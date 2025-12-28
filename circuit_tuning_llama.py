"""
Circuit-tuning
"""

from llama_recipes.configs import train_config as TRAIN_CONFIG, fsdp_config as FSDP_CONFIG
from llama_recipes.configs.datasets import custom_dataset
from llama_recipes.utils.config_utils import (
    check_fsdp_config,
)
from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)
from llama_recipes.utils.train_utils import (
    train, 
    setup, 
    clear_gpu_cache, 
    setup_environ_flags,
    freeze_transformer_layers,
    get_policies,
    
)
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from transformer_lens import HookedTransformer
from transformer_lens.components import TransformerBlockSplit

from eap.eap_graph_llama import EAPGraphLlama
from eap.eap_wrapper_llama import EAP_ablation, EAP_ablation_multi_step
from eap.patching_metrics import (
    avg_logit_diff_sv, 
    avg_log_prob_sum_math,
    avg_logit_diff_bias,
    patching_metric
)
from utils import TopNScheduler
from utils import print_rank_0, load_model_from_ckpt

import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from accelerate.utils import is_xpu_available
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.data import DataCollatorForSeq2Seq

import os
import sys
import random
import jsonlines
import argparse
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from functools import partial

# os.environ["WANDB_MODE"] = "offline"

def get_model_parameters_num(model):
    if isinstance(model, FSDP):
        with model.summon_full_params(
            module=model,
            recurse=True,
            writeback=False,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False
        ):
            model_params_num = sum(p.numel() for p in model.parameters())
            model_params_memory = sum(p.numel() * 2 / 1024 / 1024 for p in model.parameters())
    else:
        model_params_num = sum(p.numel() for p in model.parameters())
        model_params_memory = sum(p.numel() * 2 / 1024 / 1024 for p in model.parameters())
    
    return model_params_num, model_params_memory

def get_circuit_tuning_parameters_num(model, relevant_params):
    if isinstance(model, FSDP):
        with model.summon_full_params(
            module=model,
            recurse=True,
            writeback=False,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False
        ):
            real_tuning_params_num = sum(p.numel() for name, p in model.named_parameters() if name in relevant_params)
            real_tuning_params_memory = sum(p.numel() * 2 / 1024 / 1024 for name, p in model.named_parameters() if name in relevant_params)
    else:
        real_tuning_params_num = sum(p.numel() for name, p in model.named_parameters() if name in relevant_params)
        real_tuning_params_memory = sum(p.numel() * 2 / 1024 / 1024 for name, p in model.named_parameters() if name in relevant_params)
    
    return real_tuning_params_num, real_tuning_params_memory
    
    
def get_relevant_params(model, top_edges, bias=True):
    """
    Get the relevant nodes in the subgraph.
    Args:
        model: HookedTransformer
        top_edges: list of tuples
    """
    upstream_nodes = list(set([edge[0] for edge in top_edges]))
    downstream_nodes = list(set([edge[1] for edge in top_edges]))
    nodes_num = len(upstream_nodes) + len(downstream_nodes)
    
    relevant_params = []
    
    # get the edge(weight) corresponding to the node
    for node in upstream_nodes:
        # head: head.0.0 -> head.{layer}.{head_idx} (blocks.0.attn.hook_result') -> W_{O}*z + b_{O} ~ hook_result
        # mlp: mlp_head.0.0 -> mlp_head.{layer}.{head_idx}  (blocks.0.hook_mlp_out) -> W_out*gelu(pre) + b_{out} ~ hook_mlp_out
        if "head" in node:
            layer_id = node.split('.')[1]
            head_id = node.split('.')[2]
            module_name = f"blocks.{layer_id}.attn"
            param_names = ["W_O", "b_O"] if bias else ["W_O"] # (n_heads, d_head, d_model)
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}_{head_id}"
                relevant_params.append(param_key)
        elif "mlp" in node:
            layer_id = node.split('.')[1]
            head_id = node.split('.')[2]
            module_name = f"blocks.{layer_id}.mlp"
            param_names = ["W_out", "b_out"] if bias else ["W_out"]
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}_{head_id}"
                relevant_params.append(param_key)
        else:
            raise ValueError(f"Node {node} not supported")
        
    for node in downstream_nodes:
        # head: head.0.0.q -> head.{layer}.{head_idx}.{letter} (blocks.0.hook_q) -> W_{q}*x
        # mlp: mlp.0 (blocks.0.hook_mlp_in) -> W_{in}*x
        qkv_map = {"q": ["W_Q", "b_Q"], "k": ["W_K", "b_K"], "v": ["W_V", "b_V"]} if bias else {"q": ["W_Q"], "k": ["W_K"], "v": ["W_V"]}
        if "mlp" in node:
            layer_id = node.split(".")[1]
            head_id = node.split(".")[2]
            module_name = f"blocks.{layer_id}.mlp"
            param_names = ["W_in", "b_in"] if bias else ["W_in"]
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}_{head_id}"
                relevant_params.append(param_key)
        elif "head" in node:
            layer_id = node.split(".")[1]
            head_id = node.split(".")[2]
            letter = node.split(".")[3]
            module_name = f"blocks.{layer_id}.attn"
            param_names = qkv_map[letter]
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}_{head_id}"
                relevant_params.append(param_key)
        else:
            raise ValueError(f"Node {node} not supported")
    
    irrelevant_params = []
    if isinstance(model, FSDP):
        with model.summon_full_params(
            module=model,
            recurse=True,
            writeback=False,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False
        ):
            for name, _ in model.named_parameters():
                if name not in relevant_params:
                    irrelevant_params.append(name)
    else:
        for name, _ in model.named_parameters():
            if name not in relevant_params:
                irrelevant_params.append(name)
            
    return relevant_params, irrelevant_params, nodes_num
        
 
def freeze_outside_nodes(model, irrelevant_params):
    """
    We only want to update the nodes in the subgraph. So we freeze the nodes outside of the subgraph.
    """
    if isinstance(model, FSDP):
        with model.summon_full_params(
            module=model,
            recurse=True,
            writeback=False,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False
        ):
            for name, param in model.named_parameters():
                if name in irrelevant_params:
                    # print_rank_0("ok")
                    # sys.stdout.flush()
                    param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if name in irrelevant_params:
                param.requires_grad = False
           
def unfreeze_outside_nodes(model, irrelevant_params):
    
    if isinstance(model, FSDP):
        with model.summon_full_params(
            module=model,
            recurse=True,
            writeback=False,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False
        ):
            for name, param in model.named_parameters():
                if name in irrelevant_params:
                    # print_rank_0("ok")
                    # sys.stdout.flush()
                    param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if name in irrelevant_params:
                param.requires_grad = True


def random_activate(model, irrelevant_params, random_ratio=0.05):
    """
    Randomly set nodes outside of the subgraph activated. (an old version, randomly set some irrelevant params activated in every step)
    """
    added_num = 0
    added_mem = 0
    random_params = np.random.choice(irrelevant_params, int(len(irrelevant_params) * random_ratio), replace=False)
    for name, param in model.named_parameters():
        if name in random_params:
            param.requires_grad = True
            added_num += param.numel()
            added_mem += param.numel() * 2 / 1024 / 1024
    return added_num, added_mem, random_params


def random_get_topn_1000_irrelevant_params(model, random_ratio=0.02, bias=False):
    
    log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_1000-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
    
    with jsonlines.open(log_path, mode='r') as f:
        data = list(f)
    top_edges = data[-1]["edge_info"]
    
    _, irrelevant_params, _ = get_relevant_params(model, top_edges, bias)
    
    added_num = 0
    added_mem = 0
    random_params = np.random.choice(irrelevant_params, int(len(irrelevant_params) * random_ratio), replace=False)   
       
    for name, param in model.named_parameters():
        if name in random_params:
            added_num += param.numel()
            added_mem += param.numel() * 2 / 1024 / 1024
    
    return added_num, added_mem, random_params, irrelevant_params


def weighted_loss_func(logits,
                       tokens, 
                       attention_mask, 
                       verb_pos, 
                       loss_weight_p, 
                       verb_alone=False):
    """
    p(w1)p(w2)...p(verb)...p(wn) ->
    p(w1)p(w2)...p(verb)^{1/p}...p(wn), p \in (0, 1]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]

    # Ignore token positions which are masked out or where the next token is masked out
    # (generally padding tokens)
    next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
    predicted_log_probs *= next_token_mask  #  [batch_size, seq_len - 1]
    n_tokens = next_token_mask.sum().item()
    
    pre_verb_pos = [[pos - 1 for pos in data] for data in verb_pos] 
    if verb_alone:
        verb_probs = []
        for i in range(len(pre_verb_pos)):
            verb_probs.append(predicted_log_probs[i, pre_verb_pos[i]])
        loss = torch.cat([verb_prob.unsqueeze(0)  for verb_prob in verb_probs], dim=1).sum()
        total_verb_num = sum([len(data) for data in pre_verb_pos])
        loss /= total_verb_num
    else:
        for i in range(len(pre_verb_pos)):
            predicted_log_probs[i, pre_verb_pos[i]] *= 1 / loss_weight_p
        loss = -predicted_log_probs.sum() / n_tokens

    return loss
    

def circuit_tuning(
    config,
    model: HookedTransformer,
    graph: EAPGraphLlama,
    train_dataloader: DataLoader,
    rank: int,
    fsdp_config: FSDP_CONFIG
) -> HookedTransformer:
    
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.jsonl")
    
    # if config.enable_fsdp and rank == 0:
    #     all_p = []
    #     with model.summon_full_params(
    #         module=model,
    #         recurse=True,
    #         writeback=False,
    #         rank0_only=True,
    #         offload_to_cpu=True,
    #         with_grads=False
    #     ):
    #         for name, param in model.named_parameters():
    #             all_p.append(name)
    #     with jsonlines.open(log_path, mode='a') as f:
    #         f.write({"all_p": all_p})
    
    if config.wandb:
        if not config.enable_fsdp or (config.enable_fsdp and rank == 0):
            wandb.login(key=config.wandb_api_key)
            print_rank_0(config.wandb_project_name)
            print_rank_0(os.path.basename(save_dir))
            wandb.init(project=config.wandb_project_name, 
                    name=os.path.basename(save_dir))

    
    if config.optimizer_name in ["Adam", "AdamW"]:
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(config.weight_decay if config.weight_decay is not None else 0.0),
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

    if config.task == "math":
        metric_func = None if config.process_or_outcome == "process" else avg_log_prob_sum_math
    elif config.task == "bias":
        metric_func = avg_logit_diff_bias
    elif config.task == "logic":
        metric_func = None
    elif config.task == "reading":
        metric_func = None
    else:
        raise ValueError(f"Task {config.task} not supported")
    patching_metric_fn = partial(patching_metric, logit_diff_func=metric_func)
    eap_func = EAP_ablation if config.ablation_method == "zero" else EAP_ablation_multi_step
    
    if config.topn_scheduler[1] != 0:
        topn_scheduler = TopNScheduler(start_val=config.topn_scheduler[0], 
                                       end_val=config.top_n, 
                                       warmup_steps=config.topn_scheduler[1],
                                       scheduler_type=config.topn_scheduler[2])
        config.top_n = topn_scheduler.step()

    model_params_num, model_params_mem = get_model_parameters_num(model)
    print_rank_0("model_params_num", model_params_num, "model_params_mem", model_params_mem)
     
    # pre-activate some irrelevant params
    added_num, added_mem, added_param_num = 0, 0, 0
    random_params = []
    if config.random_ratio != 0:
        added_num, added_mem, random_params, _ = random_get_topn_1000_irrelevant_params(model, random_ratio=config.random_ratio, bias=config.bias)
        added_param_num = len(random_params)
    
    param_info_dict = {"real_tuning_params_num": [],
                       "real_tuning_params_mem": [],
                       "added_num": added_num,
                       "added_mem": added_mem,
                       "added_param_num": added_param_num,
                       "added_params": list(random_params),
                       "nodes_num": [],
                       "model_params_num": model_params_num,
                       "model_params_mem": model_params_mem}
    param_info_for_log = {}
    
    relevant_params, irrelevant_params = [], []
    nodes_num = 0
    top_edges = None
    edges_info = {}
    
    graph.reset_scores()
    if config.ie_over_seq:
        upstream_activations_difference  = torch.zeros(
            (config.batch_size, 1, graph.n_upstream_nodes),
            device=config.device,
            dtype=model.cfg.dtype,
            requires_grad=False
        )
    else:
        upstream_activations_difference = torch.zeros(
            (config.batch_size, graph.n_upstream_nodes),
            device=config.device,
            dtype=model.cfg.dtype,
            requires_grad=False
        )
        
    model.train()
    
    for epoch in range(1, config.num_epochs + 1):
        samples = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}", colour="green")):
                        
            if step == 0 and epoch == 1:
                print_rank_0(batch)
            
            # normal forward and backward
            logits, loss, upstream_activations_difference = eap_func(
                        model,
                        graph=graph,
                        batch=batch,
                        metric=patching_metric_fn,
                        config=config,
                        do_bwd=False,
                        upstream_activations_difference=upstream_activations_difference
                        )
            
            if config.objective == "standard" or config.loss_weight_p == 1:
                if config.task == "bias":
                    value = avg_logit_diff_bias(logits, batch)
                    loss += abs(value) * config.regularization_beta
                loss.backward()
            else:
                # for task that focus on a particular token(e.g. sv), we can add weight to the target verb
                if config.objective == "weighted" and config.task in ["sv"]:
                    loss_weight_p = config.loss_weight_p
                    weighted_loss = weighted_loss_func(logits=logits,
                                            tokens=batch["clean_inputs"]["input_ids"].to(config.device), 
                                            attention_mask=batch["clean_inputs"]["attention_mask"].to(config.device), 
                                            verb_pos=batch["clean_verb_pos"], 
                                            loss_weight_p=loss_weight_p)
                else:
                    # config.objective == "alone"
                    weighted_loss = weighted_loss_func(logits=logits, 
                                            tokens=batch["clean_inputs"]["input_ids"].to(config.device), 
                                            attention_mask=batch["clean_inputs"]["attention_mask"].to(config.device), 
                                            verb_pos=batch["clean_verb_pos"],
                                            loss_weight_p=None,
                                            verb_alone=True)
                weighted_loss.backward()
            
            
            # optimization, pruning and freezing
            if step % config.accumulation_steps == 0 or step == len(train_dataloader) - 1:
                
                # optimize
                if step > 0:
                    if config.gradient_clipping and config.max_grad_norm > 0.0:
                        if config.enable_fsdp:
                            model.clip_grad_norm_(config.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if config.topn_scheduler[1] != 0:
                        config.top_n = topn_scheduler.step()
            
                # forward and backward to get a new subgraph
                # unfreeze all nodes that were frozen before
                unfreeze_outside_nodes(model, irrelevant_params)
                graph, logits, loss = eap_func(
                        model,
                        graph=graph,
                        batch=batch,
                        metric=patching_metric_fn,
                        config=config,
                        do_bwd=True,
                        upstream_activations_difference=upstream_activations_difference
                        )
                model.zero_grad()
                if config.ie_over_seq:
                    upstream_activations_difference  = torch.zeros(
                        (config.batch_size, 1, graph.n_upstream_nodes),
                        device=config.device,
                        dtype=model.cfg.dtype,
                        requires_grad=False
                    )
                else:
                    upstream_activations_difference = torch.zeros(
                        (config.batch_size, graph.n_upstream_nodes),
                        device=config.device,
                        dtype=model.cfg.dtype,
                        requires_grad=False
                    )
                
                # prune the computation graph
                if config.threshold != 0:
                    top_edges = graph.top_edges(n=config.top_n, threshold=config.threshold, abs_scores=True, cross_layer=config.cross_layer, prune_method=config.prune_method)
                else:
                    top_edges = graph.top_edges(n=config.top_n, abs_scores=True, cross_layer=config.cross_layer, prune_method=config.prune_method)
                
                # set nodes outside of the subgraph freezed
                relevant_params, irrelevant_params, nodes_num = get_relevant_params(model, top_edges, bias=config.bias)
                freeze_outside_nodes(model, irrelevant_params)

                if config.random_ratio != 0:
                    unfreeze_outside_nodes(model, random_params)
                
                # reset graph
                graph.reset_scores()
                
                # get computation log
                real_num, real_mem = get_circuit_tuning_parameters_num(model, relevant_params)
                param_info_dict["real_tuning_params_num"].append(real_num + added_num)
                param_info_dict["real_tuning_params_mem"].append(real_mem + added_mem)
                param_info_dict["nodes_num"].append(nodes_num)
                real_param_ratio = (real_num + added_num) / model_params_num
                param_info_for_log = {"real_tuning_params_num": real_num + added_num,
                                    "real_tuning_params_mem": real_mem + added_mem,
                                    "real_param_ratio": real_param_ratio,
                                    "nodes_num": nodes_num,
                                    "added_param_num": added_param_num,
                                    "added_params": list(random_params)}
                
            # print and log
            samples += config.batch_size
            
            if config.print_every is not None and step > 0 and step % config.print_every == 0:
                print_rank_0(f"Epoch {epoch} Step {step} Loss {loss.item()} rank {rank}")
                sys.stdout.flush()

            lr = scheduler.get_last_lr()[0]
            basic_info = {"loss": loss.item(), "learning rate": lr, "step": step + (epoch - 1)*(len(train_dataloader) + 1)}
            
            # edges_info = {"topn": config.top_n, "edges": top_edges}
            edges_info = {"topn": config.top_n}

            log_info = {**basic_info, **param_info_for_log, **edges_info}
            
            # print(f"Epoch {epoch}, Step {step}, enable_fsdp: {config.enable_fsdp}, rank: {rank}")
            if not config.enable_fsdp or (config.enable_fsdp and rank == 0):
                # print_rank_0("ok")
                with jsonlines.open(log_path, mode='a') as f:
                    f.write(log_info)
                    if config.wandb:
                        wandb.log(log_info)
                        
            if epoch ==1 and ((step + 1) in [300, 400, 500, 800, 1000, 1500]):
                epoch_step = f"{step+1}steps"
                save_model(model, config, fsdp_config, epoch=epoch_step, save_dir=save_dir, rank=rank, ckpt_name=f"model-epoch_{epoch}-steps_{step+1}.pt")
            elif (step + 1) % 1000 == 0:
                save_model(model, config, fsdp_config, epoch=epoch, save_dir=save_dir, rank=rank, ckpt_name="model-epoch_{epoch}.pt")
            else:
                pass

        # lr
        scheduler.step()
        
        # save every epoch
        save_model(model, config, fsdp_config, epoch=epoch, save_dir=save_dir, rank=rank)
        
    # save subject-verb disagreement circuit
    _, edges = graph.show(edges=top_edges, fname="eap_graph_after.png", fdir=save_dir)
    
    # log final info
    final_param_info = {"avg_real_tuning_params_num": np.array(param_info_dict["real_tuning_params_num"]).mean(),
                        "avg_real_tuning_params_mem": np.array(param_info_dict["real_tuning_params_mem"]).mean(),
                        "avg_added_params_num": np.array(param_info_dict["added_num"]).mean(),
                        "avg_added_params_mem": np.array(param_info_dict["added_mem"]).mean(),
                        "avg_added_param_num": np.array(param_info_dict["added_param_num"]).mean(),
                        "avg_nodes_num": np.array(param_info_dict["nodes_num"]).mean(),
                        "model_params_num": model_params_num,
                        "model_params_mem": model_params_mem}
    final_param_info["total_nodes_num"] = graph.n_upstream_nodes + graph.n_downstream_nodes
    final_param_info["nodes_num"] = nodes_num
    final_param_info["total_edges_num"] = graph.n_upstream_nodes * graph.n_downstream_nodes
    final_param_info["edges_num"] = len(edges)
    final_param_info["edge_info"] = edges
    
    with jsonlines.open(log_path, mode='a') as f:
        f.write(final_param_info)


def load_llama_in_HookedTransformer(train_config, fsdp_config):
    """
    We firstly load llama using transformers as a huggingface style model, then pass it to HookedTransformer.
    Attention:
    (1) Precision: we use torch.bfloat16.    cfg.dtype
    (2) Device    cfg.device
    """
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if train_config.ckpt_path:
        cfg_path = os.path.join(train_config.model_path, "config.json")
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = load_model_from_ckpt(official_model_name=train_config.model_name,
                                     ckpt_path=train_config.ckpt_path,
                                     cfg_path=cfg_path,
                                     tokenizer=tokenizer,
                                     )
    else:
        use_cache = False if train_config.enable_fsdp else None
        model = LlamaForCausalLM.from_pretrained(
                train_config.model_path,
                quantization_config=None,
                use_cache=use_cache,
                attn_implementation="sdpa",
                device_map=(
                    "auto"
                    if train_config.quantization and not train_config.enable_fsdp
                    else None
                ),
                torch_dtype=torch.bfloat16,
            )
        if (
            train_config.enable_fsdp
            and fsdp_config.pure_bf16
            and not train_config.quantization
        ):
            model.to(torch.bfloat16)
        print_rank_0("Huggingface model loaded")
        
        model = HookedTransformer.from_pretrained(
                model_name=train_config.model_name,
                hf_model=model,
                tokenizer=tokenizer,
                default_padding_side="left",
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                fold_value_biases=False,
                move_to_device=False,
                dtype=torch.bfloat16,
                split_params=True,  # split head and mlp
            )
        
    if not train_config.enable_fsdp:
        model.to(train_config.device)
    # model.to(rank)  # We don't really put the model on cuda. This step is just to change the value of "cfg.device" to "cuda:rank"
        
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    
    upstream_nodes=["mlp", "head"]
    downstream_nodes=["mlp", "head"]
    graph = EAPGraphLlama(model.cfg, upstream_nodes, downstream_nodes)
    print_rank_0("Model loaded")
    
    return model, graph

def prepare_parallel_training(model, train_config, fsdp_config, rank):
    hsdp_device_mesh_plan = None
    if (
        fsdp_config.hsdp  # False
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print_rank_0("HSDP device mesh is ready")

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not train_config.use_peft and train_config.freeze_layers:  # False
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        
        # Create the FSDP wrapper for LlamaDecoderLayer in text models
        # my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
        # my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [TransformerBlockSplit])
        my_auto_wrapping_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set([TransformerBlockSplit])
    )
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        low_cpu_fsdp = False
        model = FSDP(
            model,
            auto_wrap_policy=(
                my_auto_wrapping_policy
            ),
            cpu_offload=(
                CPUOffload(offload_params=True)
                if fsdp_config.fsdp_cpu_offload
                else None
            ),
            mixed_precision=(
                mixed_precision_policy if not fsdp_config.pure_bf16 else None
            ),
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=low_cpu_fsdp,
            param_init_fn=(
                (
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if low_cpu_fsdp and rank != 0
                else None
            ),
            use_orig_params=True,
        )
        if fsdp_config.fsdp_activation_checkpointing:  # True
            # model.enable_input_require_grads()
            # model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)
            
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to(train_config.device)
    
    return model

def prepare_data(train_config, tokenizer):
    
    dataset_config = custom_dataset()
    if train_config.task == "math":
        dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k_dataset.py"
    elif train_config.task == "bias":
        dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/bias_dataset.py"
    elif train_config.task == "logic":
        dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic_dataset.py"
    elif train_config.task == "reading":
        dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad_dataset.py"
    else:
        raise ValueError(f"Task {train_config.task} not supported")
    dataset_config.train_split = "train"
    dataset_config.test_split = "test"

    dataset_train = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="train",
        )
    print_rank_0(f"--> Training Set Length = {len(dataset_train)}")
    
    if train_config.enable_fsdp:
        batch_sampler = DistributedLengthBasedBatchSampler(
            dataset_train,
            batch_size=train_config.batch_size,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
    else:
        batch_sampler = LengthBasedBatchSampler(dataset_train, train_config.batch_size, drop_last=True, shuffle=True)
    collate_func = DataCollatorForSeq2Seq(tokenizer)
    custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)
    
    if custom_data_collator:
        print_rank_0("custom_data_collator is used")
        collate_func = custom_data_collator
        
    # Create DataLoaders for the training and validation dataset
    train_dataloader = DataLoader(
        dataset_train,
        num_workers=1,
        pin_memory=True,
        batch_sampler=batch_sampler,
        collate_fn=collate_func,
    )
    print_rank_0(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")
    
    return train_dataloader

def save_model(model, train_config, fsdp_config, epoch, save_dir, rank, ckpt_name=None):
    
    def save_model_checkpoint(model, epoch, save_dir, ckpt_name=None):
        """save model when not peft and on single device"""
        
        if ckpt_name:
            output_file = os.path.join(save_dir, ckpt_name)
        else:
            output_file = os.path.join(save_dir, f"model-epoch_{epoch}.pt")
        state_dict = model.state_dict()
        
        torch.save(state_dict, output_file)
    
    def save_fsdp_model_checkpoint_full(model, rank, epoch=1, save_dir=None):
        """saving model via rank0 cpu streaming and full_state_dict"""

        fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state = model.state_dict()

            print(f"saving process: rank {rank}  done w model state_dict\n")

        if rank == 0:
            print(f"--> saving model ...")
            # create save path
            ckpt_save_dir = os.path.join(save_dir, f"fsdp-epoch_{epoch}")
            os.makedirs(ckpt_save_dir, exist_ok=True)
            
            # save model
            torch.save(cpu_state, ckpt_save_dir)
            print(f"model checkpoint saved for epoch {epoch} at {ckpt_save_dir}\n")
    
    def save_model_sharded(model, rank, epoch=1, save_dir=None):
        """save model and optimizer via sharded_state_dict to save_dir"""
    
        ckpt_save_dir = os.path.join(save_dir, f"fsdp-epoch_{epoch}")
        if rank == 0:
            print(f"Saving model to {ckpt_save_dir}")

        distributed_writer = dist_cp.FileSystemWriter(
            ckpt_save_dir,
        )

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            
            state_dict = {"model": model.state_dict()}

            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=distributed_writer,
                planner=DefaultSavePlanner(),
            )
        dist.barrier()
        if rank == 0:
            print(f"Sharded state checkpoint saved to {save_dir}")
        
      
    if train_config.enable_fsdp:
        dist.barrier()
        
    if not train_config.enable_fsdp:
        save_model_checkpoint(model, epoch, save_dir, ckpt_name)
        
    elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
        print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
        print("=====================================================")
        save_fsdp_model_checkpoint_full(model, rank, epoch, save_dir)
        
    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
        print("=====================================================")
        save_model_sharded(model, rank, epoch, save_dir)
        
    if train_config.enable_fsdp:
            dist.barrier()
    
def get_arguments():
    
    def float_or_str(value):
            try:
                return float(value)
            except ValueError:
                return value
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-instruct")
    parser.add_argument('--model_path', type=str, default="/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--bias', action='store_true', help="whether to use bias in the model")
    parser.add_argument('--task', type=str, default='math', choices=["math", "logic", "bias", "reading"])
    
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--enable_fsdp', action="store_true")
    parser.add_argument('--use_peft', action="store_true", help="We don't need this argument in circuit-tuning, just ignore it")
    parser.add_argument('--quantization', action="store_true")
    parser.add_argument('--freeze_layers', action="store_true")
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--optimizer_name', type=str, default='SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.85, help="lr decay factor")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gradient_clipping', action="store_true")
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--eval_steps', type=int, default=100)
    
    parser.add_argument('--prune_method', type=str, default="top_edges")
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--metric', type=str, default="logit_diff")
    parser.add_argument('--random_ratio', type=float, default=None)
    parser.add_argument('--topn_scheduler', type=float_or_str, nargs='+', help="topn start and topn warm up steps")
    parser.add_argument('--smooth', action='store_true', help="whether to smooth the IE scores")
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.99)
    parser.add_argument('--ablation_method', type=str, choices=["mean", "zero"], default=None)
    parser.add_argument('--ie_over_seq', action="store_true")
    parser.add_argument('--process_or_outcome', type=str, choices=["process", "outcome"], default="outcome")
    parser.add_argument('--objective', type=str, choices=["standard", "weighted", "alone"], default="standard")
    parser.add_argument('--loss_weight_p', type=float, default=1.0)
    parser.add_argument('--regularization_beta', type=float, default=0)
    parser.add_argument('--cross_layer', action='store_true')
    
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_api_key', type=str, default=None)
    parser.add_argument('--wandb_project_name', type=str, default=None)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)  # /checkpoints/xxx
    
    args = parser.parse_args()
    print(args)
    
    return args

def train():
    """
    Circuit-tuning
    """
    
    print("Circuit-tuning")
    # ===================== Get arguments =====================
    args = get_arguments()
    
    fsdp_config = FSDP_CONFIG()
    fsdp_config.pure_bf16 = True
    fsdp_config.fsdp_cpu_offload = False

    # ===================== Prepare environment =====================
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        args.device = local_rank
    else:
        rank = args.device.split(':')[-1]

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    # ===================== Load model =====================
    print_rank_0('Loading model...')
            
    model, graph = load_llama_in_HookedTransformer(args, fsdp_config)
    
    # ===================== Prepare parallel training =====================
    if args.enable_fsdp:
        print_rank_0('Preparing parallel training...')
        model = prepare_parallel_training(model, args, fsdp_config, rank)
        # import pdb; pdb.set_trace()

    
    # ==================== Prepare data ====================
    print_rank_0('Preparing data...')
    train_dataloader = prepare_data(args, model.tokenizer)
        
    # ==================== Start training ====================
    circuit_tuning(config=args, model=model, graph=graph, train_dataloader=train_dataloader, rank=rank, fsdp_config=fsdp_config)
    print('Training done!')


if __name__ == '__main__':
    
    def func():
        """
        Simple test
        """
        model = HookedTransformer.from_pretrained(
            'gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda:2"
        )
        # import pdb; pdb.set_trace()
        added_num, added_mem, random_params, params_to_choose = random_get_topn_1000_irrelevant_params(model, 'gpt2-small', random_ratio=0.06, bias=False)
        # 68 * ratio
        print(len(random_params), len(params_to_choose), random_params)
        
    def test():
        pass
        
    train()
    
    
    
    