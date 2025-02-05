"""
Circuit-tuning
"""

import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils
from transformer_lens.train import train
from eap.eap_graph_old import EAPGraph
from eap.eap_wrapper import EAP_standard, EAP_ablation
from eap.patching_metrics import (
    avg_logit_diff_sv, 
    avg_neg_log_prob_diff_sv,
    avg_logit_diff_bias,
    patching_metric
)
from circuit_data import SVDataset, SVCollateFn, BiasDataset, BiasCollateFn
from eval.sv.sv import evaluate_sv
from eval.bias.bias import evaluate_bias
from utils import TopNScheduler

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

import argparse
from tqdm import tqdm
import wandb
from typing import Optional
import os
import sys
from functools import partial
from copy import deepcopy

import logging
import jsonlines

# os.environ["WANDB_MODE"] = "offline"

def get_model_parameters_num(model):
    model_params_num = sum(p.numel() for p in model.parameters())
    model_params_memory = sum(p.numel() * 2 / 1024 / 1024 for p in model.parameters())
    return model_params_num, model_params_memory

def get_circuit_tuning_parameters_num(model, relevant_params):
    real_tuning_params_num = sum(p.numel() for name, p in model.named_parameters() if name in relevant_params)
    real_tuning_params_memory = sum(p.numel() * 2 / 1024 / 1024 for name, p in model.named_parameters() if name in relevant_params)
    ideal_tuning_params_num = real_tuning_params_num
    ideal_tuning_params_memory = real_tuning_params_memory
    for name, p in model.named_parameters():
        if name in relevant_params:
            if relevant_params[name]["slice"] is not None:
                for slice in relevant_params[name]["slice"]:
                    ideal_tuning_params_num -= p[slice].numel()
                    ideal_tuning_params_memory -= p[slice].numel() * 2 / 1024 / 1024
    
    return real_tuning_params_num, real_tuning_params_memory, ideal_tuning_params_num, ideal_tuning_params_memory
    
    
def get_relevant_params(model, model_name, top_edges, bias=True):
    """
    Get the relevant nodes in the subgraph.
    Args:
        model: HookedTransformer
        model_name: str. Used for deciding the relevant params -> Llama series don't have bias params.
        top_edges: list of tuples
    """
    upstream_nodes = list(set([edge[0] for edge in top_edges]))
    downstream_nodes = list(set([edge[1] for edge in top_edges]))
    nodes_num = len(upstream_nodes) + len(downstream_nodes)
    
    relevant_params = {}
    
    # get the edge(weight) corresponding to the node
    for node in upstream_nodes:
        # head: head.0.0 (blocks.0.attn.hook_result') -> W_{O}*z + b_{O}
        # mlp: mlp.0 (blocks.0.hook_mlp_out) -> W_out*gelu(pre) + b_{out}
        if "head" in node:
            layer_id = node.split(".")[1]
            head_id = node.split(".")[2]
            module_name = f"blocks.{layer_id}.attn"
            param_names = ["W_O", "b_O"] if bias else ["W_O"] # (n_heads, d_head, d_model)
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}"
                relevant_params[param_key] = {"slice": [slice(0, int(head_id)), slice(int(head_id)+1, None)], "pos": 0}
        elif "mlp" in node:
            layer_id = node.split(".")[1]
            module_name = f"blocks.{layer_id}.mlp"
            param_names = ["W_out", "b_out"] if bias else ["W_out"]
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}"
                relevant_params[param_key] = {"slice": None}
        else:
            raise ValueError(f"Node {node} not supported")
        
    for node in downstream_nodes:
        # head: head.0.0.q (blocks.0.hook_q_input) -> W_{q}*x
        # mlp: mlp.0 (blocks.0.hook_mlp_in) -> W_{in}*x
        qkv_map = {"q": ["W_Q", "b_Q"], "k": ["W_K", "b_K"], "v": ["W_V", "b_V"]} if bias else {"q": ["W_Q"], "k": ["W_K"], "v": ["W_V"]}
        if "head" in node:
            layer_id = node.split(".")[1]
            head_id = node.split(".")[2]
            letter = node.split(".")[3]
            module_name = f"blocks.{layer_id}.attn"
            param_names = qkv_map[letter]
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}"
                relevant_params[param_key] = {"slice": None}
        elif "mlp" in node:
            layer_id = node.split(".")[1]
            module_name = f"blocks.{layer_id}.mlp"
            param_names = ["W_in", "b_in"] if bias else ["W_in"]
            for param_name in param_names:
                param_key = f"{module_name}.{param_name}"
                relevant_params[param_key] = {"slice": None}
        else:
            raise ValueError(f"Node {node} not supported")
    
    irrelevant_params = []
    for name, _ in model.named_parameters():
        if name not in relevant_params:
            irrelevant_params.append(name)
            
    return relevant_params, irrelevant_params, nodes_num
        
 
def freeze_outside_nodes(model, irrelevant_params):
    """
    We only want to update the nodes in the subgraph. So we freeze the nodes outside of the subgraph.
    """
    for name, param in model.named_parameters():
        if name in irrelevant_params:
            param.requires_grad = False
           
def unfreeze_outside_nodes(model, irrelevant_params):
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


def random_get_topn_1000_irrelevant_params(model, model_name, random_ratio=0.02, bias=True):
    
    log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_1000-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
    
    with jsonlines.open(log_path, mode='r') as f:
        data = list(f)
    top_edges = data[-1]["edge_info"]
    
    _, irrelevant_params, _ = get_relevant_params(model, model_name, top_edges, bias)
    
    added_num = 0
    added_mem = 0
    random_params = []
    # interest_params_list = ["W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O", "b_O", "W_in", "b_in", "W_out", "b_out"]
    
    # passed = False
    # while not passed:
    #     random_params = np.random.choice(irrelevant_params, int(len(irrelevant_params) * random_ratio), replace=False)
    #     passed = True
    #     for param in random_params:
    #         if not any([name in param for name in interest_params_list]):
    #             passed = False
    #             break   
    
    # params_to_choose = deepcopy(irrelevant_params)
    # print(params_to_choose)
    # params_to_choose = [param for param in params_to_choose if any([name in param for name in interest_params_list])]
    
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
    

def train(
    config,
    model: HookedTransformer,
    graph: EAPGraph,
) -> HookedTransformer:
    
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.jsonl")
    
    torch.manual_seed(config.seed)
    
    if config.wandb:
        wandb.login(key=config.wandb_api_key)
        print(config.wandb_project_name)
        print(os.path.basename(save_dir))
        wandb.init(project=config.wandb_project_name, 
                   name=os.path.basename(save_dir), 
                   settings=wandb.Settings(start_method="fork"))

    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
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

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )
    
    if config.task == "sv":
        train_dataset = SVDataset(model.tokenizer, "train")
        dev_dataset = SVDataset(model.tokenizer, "val")
    elif config.task == "bias":
        train_dataset = BiasDataset(model.tokenizer)
    else:
        raise ValueError(f"Task {config.task} not supported")
    
    train_steps_per_epoch = len(train_dataset) // config.batch_size
    save_steps = config.save_every if config.save_every is not None else train_steps_per_epoch - 1

    if config.task == "sv":
        metric_func = avg_logit_diff_sv if "logit" in config.metric else avg_neg_log_prob_diff_sv
    elif config.task == "bias":
        metric_func = avg_logit_diff_bias
    else:
        raise ValueError(f"Task {config.task} not supported")
    patching_metric_fn = partial(patching_metric, logit_diff_func=metric_func)
        
    if config.task == "sv":
        # subject-verb agreement
        sv_collate_fn = SVCollateFn(model.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=sv_collate_fn)
        eval_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=sv_collate_fn)
        eval_func = evaluate_sv
    elif config.task == "bias":
        bias_collate_fn = BiasCollateFn(model.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=bias_collate_fn)
        eval_func = evaluate_bias
    else:
        raise ValueError(f"Task {config.task} not supported")
        
    eap_func = EAP_ablation if config.ablation_method else EAP_standard
  
    if config.topn_scheduler[1] != 0:
        topn_scheduler = TopNScheduler(start_val=config.topn_scheduler[0], 
                                       end_val=config.top_n, 
                                       warmup_steps=config.topn_scheduler[1],
                                       scheduler_type=config.topn_scheduler[2])
        config.top_n = topn_scheduler.step()
    
    model_params_num, model_params_mem = get_model_parameters_num(model)
    print(model_params_num, model_params_mem)
    
    top_edges = None
    edges_before = None
    
    model.train()
    
    # pre-activate some irrelevant params
    added_num, added_mem, added_param_num = 0, 0, 0
    random_params = []
    if config.random_ratio != 0:
        added_num, added_mem, random_params, _ = random_get_topn_1000_irrelevant_params(model, config.model, random_ratio=config.random_ratio, bias=config.bias)
        added_param_num = len(random_params)
    
    param_info_dict = {"real_tuning_params_num": [],
                       "real_tuning_params_mem": [],
                       "ideal_tuning_params_num": [],
                       "ideal_tuning_params_mem": [],
                       "added_num": added_num,
                       "added_mem": added_mem,
                       "added_param_num": added_param_num,
                       "added_params": list(random_params),
                       "nodes_num": [],
                       "model_params_num": model_params_num,
                       "model_params_mem": model_params_mem}
    
    
    for epoch in range(1, config.num_epochs + 1):
        samples = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            
            # find subject-verb agreement circuit
            if step == 0 and epoch == 1:
                for key, value in train_dataset[0].items():
                    print(key, value)
                    sys.stdout.flush()
                    
                graph.reset_scores()
                
                if config.task == "sv":
                    batch = {
                        "clean_inputs": batch["corr_inputs"],
                        "corr_inputs": batch["clean_inputs"],
                        "clean_verb_pos": batch["corr_verb_pos"],
                        "corr_verb_pos": batch["clean_verb_pos"],
                        "clean_verb_ids": batch["corr_verb_ids"],
                        "corr_verb_ids": batch["clean_verb_ids"],
                    }  # sv agreement
                print("batch:", batch)
                graph, _, _ = eap_func(
                    model,
                    graph=graph,
                    batch=batch,
                    metric=patching_metric_fn,
                    config=config,
                    )
                if config.threshold != 0:
                    top_edges = graph.top_edges(n=config.top_n, threshold=config.threshold, abs_scores=True, cross_layer=config.cross_layer)
                else:
                    top_edges = graph.top_edges(n=config.top_n, abs_scores=True, cross_layer=config.cross_layer)
                print("top_edges_max:", top_edges[0], "num_edges", len(top_edges))
                
                _, edges_before = graph.show(edges=top_edges, fname="eap_graph_before.png", fdir=config.save_dir)
                graph.reset_scores()
            
            # forward and get subgraph
            graph, logits, loss = eap_func(
                model,
                graph=graph,
                batch=batch,
                metric=patching_metric_fn,
                config=config,
                )
            if config.threshold != 0:
                top_edges = graph.top_edges(n=config.top_n, threshold=config.threshold, abs_scores=True, cross_layer=config.cross_layer)
            else:
                top_edges = graph.top_edges(n=config.top_n, abs_scores=True, cross_layer=config.cross_layer)
            print("top_edges_max:", top_edges[0], "num_edges", len(top_edges))
            sys.stdout.flush()
            
            # set nodes outside of the subgraph freezed
            relevant_params, irrelevant_params, nodes_num = get_relevant_params(model, config.model, top_edges, config.bias)
            freeze_outside_nodes(model, irrelevant_params)
            
            if config.random_ratio != 0:
                unfreeze_outside_nodes(model, random_params)
            
            # get computation
            real_num, real_mem, ideal_num, ideal_mem = get_circuit_tuning_parameters_num(model, relevant_params)
            param_info_dict["real_tuning_params_num"].append(real_num + added_num)
            param_info_dict["real_tuning_params_mem"].append(real_mem + added_mem)
            param_info_dict["ideal_tuning_params_num"].append(ideal_num + added_num)
            param_info_dict["ideal_tuning_params_mem"].append(ideal_mem + added_mem)
            param_info_dict["nodes_num"].append(nodes_num)
            real_param_ratio = (real_num + added_num) / model_params_num
            ideal_param_ratio = (ideal_num + added_num) / model_params_num
            param_info_for_log = {"real_tuning_params_num": real_num + added_num,
                                  "real_tuning_params_mem": real_mem + added_mem,
                                  "ideal_tuning_params_num": ideal_num + added_num,
                                  "ideal_tuning_params_mem": ideal_mem + added_mem,
                                  "real_param_ratio": real_param_ratio,
                                  "ideal_param_ratio": ideal_param_ratio,
                                  "nodes_num": nodes_num,
                                  "added_param_num": added_param_num,
                                  "added_params": list(random_params)}
                       
            # backward and mask grads
            if config.objective == "standard" or config.loss_weight_p == 1:
                if config.task in ["sv", "bias"]:
                    loss.backward()
                else:
                    pass
            else:
                # for sv task, we can add weight to the target verb
                if config.objective == "weighted" and config.task in ["sv"]:
                    loss_weight_p = config.loss_weight_p
                    weighted_loss = weighted_loss_func(logits=logits,
                                              tokens=batch["clean_inputs"]["input_ids"].to(config.device), 
                                              attention_mask=batch["clean_inputs"]["attention_mask"].to(config.device), 
                                              verb_pos=batch["clean_verb_pos"], 
                                              loss_weight_p=loss_weight_p)
                else:
                    weighted_loss = weighted_loss_func(logits=logits, 
                                              tokens=batch["clean_inputs"]["input_ids"].to(config.device), 
                                              attention_mask=batch["clean_inputs"]["attention_mask"].to(config.device), 
                                              verb_pos=batch["clean_verb_pos"],
                                              loss_weight_p=None,
                                              verb_alone=True)
                weighted_loss.backward()
 
            for name, param in model.named_parameters():
                if name in relevant_params and relevant_params[name]["slice"] is not None:
                    # set irrelevant heads to zero
                    # W_{O}
                    for slice in relevant_params[name]["slice"]:
                        param.grad[slice] = torch.zeros_like(param.grad[slice])
            
            # optimize
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if config.accumulation_steps > 1:
                if (step + 1) % config.accumulation_steps == 0:
                    optimizer.step()
                    if config.warmup_steps > 0:
                        assert scheduler is not None
                        scheduler.step()
                    optimizer.zero_grad()
                    
                    if config.topn_scheduler[1] != 0:
                        config.top_n = topn_scheduler.step()
            else:
                optimizer.step()
                if config.warmup_steps > 0:
                    assert scheduler is not None
                    scheduler.step()
                optimizer.zero_grad()
                
                if config.topn_scheduler[1] != 0:
                    config.top_n = topn_scheduler.step()
            
            # unfreeze all nodes
            unfreeze_outside_nodes(model, irrelevant_params)
            
            # reset graph
            graph.reset_scores()

            # print and log
            samples += config.batch_size
            
            if config.print_every is not None and step > 0 and step % config.print_every == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item()}")
                sys.stdout.flush()

            lr = scheduler.get_last_lr()[0]
            
            # evaluation
            eval_info = {}
            if config.eval_steps is not None and step > 0 and step % config.eval_steps == 0:
                
                # evaluate on dev set
                if config.task == "sv":
                    eval_result = eval_func(model, eval_dataloader, config)
                elif config.task == "bias":
                    eval_result = eval_func(model, model.tokenizer, device=config.device)
                else:
                    raise ValueError(f"Task {config.task} not supported")
                eval_info = {"eval_result": eval_result}
            
            # save edges info
            edges_info = {}
            if config.eval_steps is not None and step > 0 and step % config.eval_steps == 0:
                edges_info = {"step": step + (epoch - 1)*(save_steps + 1), "edges": top_edges}
            
            basic_info = {"loss": loss.item(), "learning rate": lr, "topn": config.top_n}
            log_info = {**basic_info, **param_info_for_log, **eval_info, **edges_info}
            with jsonlines.open(log_path, mode='a') as f:
                f.write(log_info)
            if config.wandb:
                wandb.log(log_info)
            
            # save
            if step > 0 and step % save_steps == 0:
                save_path = os.path.join(save_dir, f"model-steps_{step}_epoch_{epoch}.pt")
                torch.save(model.state_dict(), save_path)

            if config.max_steps is not None and step >= config.max_steps:
                break
            
    # save subject-verb disagreement circuit
    _, edges = graph.show(edges=top_edges, fname="eap_graph_after.png", fdir=save_dir)
    
    # log final info
    final_param_info = {"avg_real_tuning_params_num": np.array(param_info_dict["real_tuning_params_num"]).mean(),
                        "avg_real_tuning_params_mem": np.array(param_info_dict["real_tuning_params_mem"]).mean(),
                        "avg_ideal_tuning_params_num": np.array(param_info_dict["ideal_tuning_params_num"]).mean(),
                        "avg_ideal_tuning_params_mem": np.array(param_info_dict["ideal_tuning_params_mem"]).mean(),
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
    final_param_info["top_edges_before"] = edges_before
    
    with jsonlines.open(log_path, mode='a') as f:
        f.write(final_param_info)

    return model, edges


if __name__ == '__main__':
    
    def func_0():
        print("Circuit-tuning")
        # get training args
        
        def float_or_str(value):
            try:
                return float(value)
            except ValueError:
                return value
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='gpt2-small')
        parser.add_argument('--bias', action='store_true', help="whether to use bias in the model")
        parser.add_argument('--task', type=str, default='sv')
        parser.add_argument('--device', type=str,  default="cpu")
        
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--optimizer_name', type=str, default='SGD')
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_steps', type=int, default=200)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--max_grad_norm', type=float, default=None)
        parser.add_argument('--accumulation_steps', type=int, default=4)
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
        parser.add_argument('--beta_2', type=float, default=0.9)
        parser.add_argument('--ablation_method', type=str, choices=["mean", "zero"], default=None)
        parser.add_argument('--ie_over_seq', action="store_true")
        parser.add_argument('--process_or_outcome', type=str, choices=["process", "outcome"], default="outcome")
        parser.add_argument('--objective', type=str, choices=["standard", "weighted", "alone"], default="standard")
        parser.add_argument('--loss_weight_p', type=float, default=1.0)
        parser.add_argument('--regularization_beta', type=float, default=0)
        parser.add_argument('--cross_layer', action='store_true')
        
        parser.add_argument('--seed', type=int, default=14)
        parser.add_argument('--wandb', action='store_true')
        parser.add_argument('--wandb_api_key', type=str, default=None)
        parser.add_argument('--wandb_project_name', type=str, default=None)
        parser.add_argument('--print_every', type=int, default=10)
        parser.add_argument('--save_every', type=int, default=None)
        parser.add_argument('--save_dir', type=str, default=None)  # /checkpoints/xxx
        
        args = parser.parse_args()
        print(args)
        
        # get model
        print('Loading model...')
                
        model = HookedTransformer.from_pretrained(
            model_name=args.model,
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device=args.device
        )
        print("model.cfg.device:", model.cfg.device)
        
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
        
        # train
        _, edges_after = train(config=args, model=model, graph=graph)
        print('Training done!')
    
    def func():   
        model = HookedTransformer.from_pretrained(
            'gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda:2"
        )
        # import pdb; pdb.set_trace()
        added_num, added_mem, random_params, params_to_choose = random_get_topn_1000_irrelevant_params(model, 'gpt2-small', random_ratio=0.06, bias=True)
        # 68 * ratio
        print(len(random_params), len(params_to_choose), random_params)
        
    func_0()
    
    
    
    