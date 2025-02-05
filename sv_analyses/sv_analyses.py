"""
Script for analyzing the subject-verb agreement task.
"""

from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model_from_ckpt

import numpy as np
import dataclasses
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jsonlines
from tqdm import tqdm

# @dataclasses.dataclass
# class Config:
#     device: str = "cuda:3"
#     beta_1: float = 0.9
#     beta_2: float = 0.9
#     task: str = "sv"
#     ie_over_seq: bool = False
#     process_or_outcome: str = "outcome"
#     ablation_method: str = "mean"
#     smooth: bool = False

# config = Config()

def load_model(model_name, ckpt_path, device):
    cfg_path = "/home/lyy/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json"
    if ckpt_path:
        model = load_model_from_ckpt(model_name,
                                     ckpt_path,
                                     cfg_path)
    else:
        model = HookedTransformer.from_pretrained(
            'gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device=device
        )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    
    model.to(device)
    
    return model

def get_hooks_from_nodes(upstream_nodes, downstream_nodes, info=[]):
    upstream_node_hook_map = {}
    downstream_node_hook_map = {}
    
    if upstream_nodes:
        for node in upstream_nodes:
            node_type = node.split(".")[0] #  'mlp' or 'head'
            layer = int(node.split(".")[1])
            if node_type == "head":
                if "pattern" in info:
                    hook_type = "attn.hook_pattern"
                else:
                    hook_type = "attn.hook_result"
            else:
                hook_type = "hook_mlp_out"
            upstream_node_hook_map[node] = f"blocks.{layer}.{hook_type}"
        
    if downstream_nodes:
        for node in downstream_nodes:
            node_type = node.split(".")[0] # 'mlp' or 'head'
            layer = int(node.split(".")[1])

            if node_type == "mlp":
                hook_type = "hook_mlp_in"
                downstream_node_hook_map[node] = f"blocks.{layer}.{hook_type}"
            elif node_type == "head":
                letter = node.split(".")[3]
                downstream_node_hook_map[node] = f"blocks.{layer}.hook_{letter}_input"
            else:
                raise NotImplementedError("Invalid downstream node")

    return upstream_node_hook_map, downstream_node_hook_map

def get_heads_for_logit_lens_no_ln(model):
    
    upstream_nodes = []
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            upstream_nodes.append(f"head.{i}.{j}")
    
    upstream_node_hook_map, downstream_node_hook_map = get_hooks_from_nodes(upstream_nodes, None)
    upstream_hooks = list(set(upstream_node_hook_map.values()))
    downstream_hooks = list(set(downstream_node_hook_map.values()))
    all_hooks = upstream_hooks + downstream_hooks
    print(all_hooks)
    
    upstream_node_values = {}
    downstream_node_values = {}
    
    def fetch_head_logitlens_prods(activation, hook, logit_lens, pre_word_positions):
        
        if hook.name in upstream_node_hook_map.values():
            # head.4.4 (blocks.4.hook_result): (bsz, len, n_head, d_model)
            # mlp.4. (blocks.4.hook_mlp_out): (bsz, len, d_model)
            for key, val in upstream_node_hook_map.items():
                if val == hook.name:
                    if "mlp" in key:
                        upstream_node_values[key] = activation.cpu()
                    else:
                        head_id = int(key.split(".")[2])
                        head_act = activation[:, :, head_id, :] # (bsz, len, d_model)
                        prods = []
                        for i in range(len(pre_word_positions)):
                            head_act_at_pre_word = head_act[i, pre_word_positions[i], :] # (d_model)
                            inner_prod = torch.matmul(head_act_at_pre_word, logit_lens[i].T)
                            prods.append(inner_prod)
                        upstream_node_values[key] = torch.stack(prods, dim=0).mean(dim=0)
        else:
            # head.4.4.q (blocks.4.hook_q): (bsz, len, n_head, d_model)
            # mlp.4 (blocks.4.hook_pre): (bsz, len, d_model)
            for key, val in downstream_node_hook_map.items():
                if val == hook.name:
                    if "mlp" in key:
                        downstream_node_values[key] = activation.cpu()
                    else:
                        head_id = int(key.split(".")[2])
                        downstream_node_values[key] = activation[:, :, head_id, :].cpu()
              
    # get logit lens
    text = [
        "I am a teacher.",
        "The weather is great.",
        "He is going to fix the car.",
        "You are a good boy.",
    ]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [0, 0, 0, 0]
    verbs = ["am", "is", "is", "are"]
    anti_verbs = ["is", "are", "are", "am"]
    
    logit_lens = []
    for i in range(len(text)):
        verb_id = model.tokenizer.tokenize(' ' + verbs[i])[0]
        anti_verb_id = model.tokenizer.tokenize(' ' + anti_verbs[i])[0]
        verb_id = model.tokenizer.convert_tokens_to_ids(verb_id)
        anti_verb_id = model.tokenizer.convert_tokens_to_ids(anti_verb_id)
        logit_diff = model.W_U[:, verb_id] - model.W_U[:, anti_verb_id]
        logit_lens.append(logit_diff)
    
    hook_func = partial(fetch_head_logitlens_prods, 
                        logit_lens=logit_lens, 
                        pre_word_positions=pre_verb_word_positions)
    fwd_hooks = [(hook, hook_func) for hook in all_hooks]
    
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                return_type="loss",
                fwd_hooks=fwd_hooks
            )
        
    topn = 10
    vals = list(upstream_node_values.values())
    vals.sort(key=lambda x: x.item(), reverse=True)
    top_vals = vals[:topn]
    print(top_vals)
    # find the corresponding keys
    keys = []
    for val in top_vals:
        for key, value in upstream_node_values.items():
            if value == val:
                keys.append(key)
                break
    print(keys)
    
def get_heads_for_logit_lens(sv_mode="sv"):
    
    upstream_nodes = []
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            upstream_nodes.append(f"head.{i}.{j}")
    
    upstream_node_hook_map, downstream_node_hook_map = get_hooks_from_nodes(upstream_nodes, None)
    upstream_hooks = list(set(upstream_node_hook_map.values()))
    downstream_hooks = list(set(downstream_node_hook_map.values()))
    all_hooks = upstream_hooks + downstream_hooks
    print(all_hooks)
    
    upstream_node_values = {}
    downstream_node_values = {}
        
    def fetch_activations(activation, hook, pre_word_positions):
    
        if hook.name in upstream_node_hook_map.values():
            # head.4.4 (blocks.4.hook_result): (bsz, len, n_head, d_model)
            # mlp.4. (blocks.4.hook_mlp_out): (bsz, len, d_model)
            for key, val in upstream_node_hook_map.items():
                if val == hook.name:
                    if "mlp" in key:
                        upstream_node_values[key] = activation.cpu()
                    else:
                        head_id = int(key.split(".")[2])
                        head_act = activation[:, :, head_id, :] # (bsz, len, d_model)
                        acts = []
                        for i in range(len(pre_word_positions)):
                            head_act_at_pre_word = head_act[i, pre_word_positions[i], :] # (d_model)
                            acts.append(head_act_at_pre_word)
                        upstream_node_values[key] = torch.stack(acts, dim=0)  # (bsz, d_model)
        else:
            # head.4.4.q (blocks.4.hook_q): (bsz, len, n_head, d_model)
            # mlp.4 (blocks.4.hook_pre): (bsz, len, d_model)
            for key, val in downstream_node_hook_map.items():
                if val == hook.name:
                    if "mlp" in key:
                        downstream_node_values[key] = activation.cpu()
                    else:
                        head_id = int(key.split(".")[2])
                        downstream_node_values[key] = activation[:, :, head_id, :].cpu()
              
    # get logit lens
    text = [
        "I am a teacher.",
        "The weather is great.",
        "He is going to fix the car.",
        "You are a good boy.",
        "I want to go to school.",
        "She likes reading books.",
        "Tom and Jerry are good friends.",
        "He goes to school by bike.",
    ]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [0, 0, 0, 0, 0, 0, 0, 0]
    verbs = ["am", "is", "is", "are", "want", "likes", "are", "goes"]
    anti_verbs = ["is", "are", "are", "am", "want", "dislikes", "is", "go"]
    
    # ===============================
    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = [item for item in lines if len(item["clean_verb_pos"]) == 1]
    
    
    sample_num = 512
    lines = lines[:sample_num]
    text = [item["clean_text"] for item in lines]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [item["clean_verb_pos"][0] - 1 for item in lines]
    verb_ids = [item["clean_verb_ids"][0] for item in lines]
    anti_verb_ids = [item["corr_verb_ids"][0] for item in lines]
    
    logit_lens = []
    for i in range(len(text)):
        # verb_id = model.tokenizer.tokenize(' ' + verbs[i])[0]
        # anti_verb_id = model.tokenizer.tokenize(' ' + anti_verbs[i])[0]
        # verb_id = model.tokenizer.convert_tokens_to_ids(verb_id)
        # anti_verb_id = model.tokenizer.convert_tokens_to_ids(anti_verb_id)
        # logit_diff = model.W_U[:, verb_id] - model.W_U[:, anti_verb_id]
        logit_diff = model.W_U[:, verb_ids[i]] - model.W_U[:, anti_verb_ids[i]]
        logit_lens.append(logit_diff)
    
    hook_func = partial(fetch_activations,
                        pre_word_positions=pre_verb_word_positions)
    fwd_hooks = [(hook, hook_func) for hook in all_hooks]
    
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                return_type="loss",
                fwd_hooks=fwd_hooks
            )
    
    head_logitlens_prods = {}
    with torch.no_grad():
        for key, value in upstream_node_values.items():
            # layer_id = int(key.split(".")[1])
            # ln_name = f"blocks.{layer_id}.ln1_post"
            # ln_layer = getattr(model, ln_name)
            head_act_at_pre_word_after_ln = model.ln_final(value)  # (bsz, d_model)
            prods = []
            for i in range(len(pre_verb_word_positions)):
                head_act = head_act_at_pre_word_after_ln[i, :]
                prod = torch.matmul(head_act, logit_lens[i].T)
                prods.append(prod)
            head_logitlens_prods[key] = torch.stack(prods, dim=0).mean(dim=0).cpu()
        
        
    topn = 20
    vals = list(head_logitlens_prods.values())
    vals.sort(key=lambda x: x.item(), reverse=True)
    top_vals = vals[:topn]
    print(top_vals)
    # find the corresponding keys
    keys = []
    for val in top_vals:
        for key, value in head_logitlens_prods.items():
            if value == val:
                keys.append(key)
                break
    print(keys)
    
    # draw graph
    data = np.array([[head_logitlens_prods[f"head.{i}.{j}"]  for j in range(model.cfg.n_heads)] for i in range(model.cfg.n_layers)])
    plt.figure(figsize=(10, 8))
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vmax = data.max(), vcenter=0)
    plt.imshow(data, cmap=plt.cm.PiYG, aspect='auto', norm=norm)  # PiYG(light green, violet) 发散调色板
    plt.colorbar()

    plt.xlabel('Attention Head')
    plt.ylabel('Layer')

    plt.xticks(np.arange(model.cfg.n_heads))
    plt.yticks(np.arange(model.cfg.n_layers))
    
    plt.tight_layout()

    task = "subject-verb agreement" if sv_mode == "sv" else "subject-verb disagreement"
    plt.title('Inner product between attention head ouputs and logit difference\nin the {} task'.format(task))
    if sv_mode == "sv":
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/sv"
    else:
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/svd"
    save_path = os.path.join(save_dir, f"head_logitlens_prods_sample{sample_num}.pdf")
    plt.savefig(save_path)

def check_single_attention_pattern(model, sv_mode="sv"):
    
    head_subj_number_sv = ["head.8.5", "head.10.9"]
    head_subj_number_svd = ["head.6.0", "head.5.2", "head.4.3"]
    if sv_mode == "sv":
        upstream_nodes = head_subj_number_sv
    else:
        upstream_nodes = head_subj_number_svd
        
    upstream_nodes = []
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            upstream_nodes.append(f"head.{i}.{j}")
    
    upstream_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=upstream_nodes, downstream_nodes=None, info=["pattern"])
    upstream_hooks = list(set(upstream_node_hook_map.values()))
    all_hooks = upstream_hooks
    print(all_hooks)
    
    # text = "When John and Mary went to a store, John gave a bottle of water to Mary."
    text = "We apologize, but this video has failed to load."
    text = "But every now and then, the interaction between opponents presents humorous or eye-opening snapshots."
    text = "Every now and then, the interaction between opponents presents humorous to us."
    text = "He didn't expect his recovery to be as long as it"
    text = "Most trading on financial markets is people buying something to try to sell it for more money later."
    text_inputs = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    
    upstream_node_values = {}
    
    def fetch_activations(activation, hook):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in upstream_node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                upstream_node_values[key] = activation[:, head_id, :, :].squeeze(0).cpu()
                        
    fwd_hooks = [(hook, fetch_activations) for hook in all_hooks]
    
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                return_type="loss",
                fwd_hooks=fwd_hooks
            )
    # plot attention pattern for each head
    save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/sv/pattern"
    save_dir = os.path.join(save_dir, "trading is")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for key, value in upstream_node_values.items():
        tokens = model.tokenizer.tokenize(text)
        tokens = [token.strip('Ġ') for token in tokens]
        plt.figure(figsize=(10, 9))
        plt.imshow(value, cmap=plt.cm.Blues, aspect='auto')
        plt.colorbar()
        # write tokens
        plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
        plt.yticks(np.arange(len(tokens)), tokens)
        plt.tick_params(axis='x', labeltop=True, labelbottom=False)
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.gca().xaxis.set_label_position('top')
        plt.title(f'Attention pattern of {key}')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{key}.pdf")
        plt.savefig(save_path)
    
    return upstream_node_values

def check_attention_at_subj(sv_mode="sv", data_type=None):
    """
    Check for all heads the attention pattern at the END token back to the subject.
    """
    
    head_subj_number_sv = ["head.8.5", "head.10.9"]
    head_subj_number_svd = ["head.6.0", "head.5.2", "head.4.3"]
    
    # if sv_mode == "sv":
    #     upstream_nodes = head_subj_number_sv
    # else:
    #     upstream_nodes = head_subj_number_svd
    
    upstream_nodes = []    
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            upstream_nodes.append(f"head.{i}.{j}")
    
    upstream_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=upstream_nodes, downstream_nodes=None, info=["pattern"])
    upstream_hooks = list(set(upstream_node_hook_map.values()))
    all_hooks = upstream_hooks
    print(all_hooks)
    
    if data_type == '0':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
    elif data_type == '1':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
    elif data_type == "mix":
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = lines[:] if data_type is not None else lines[:256]
    
    text = [item["clean_text"] for item in lines]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [item["clean_verb_pos"][0] - 1 for item in lines]
    subj_ids = [item["subj_pos"] for item in lines]
    
    upstream_node_values = {}
    
    def fetch_activations(activation, hook, pre_word_positions, subj_positions):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in upstream_node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                pattern = []
                for i in range(activation.shape[0]):
                    pattern.append(activation[i, head_id, pre_word_positions[i], subj_positions[i]].sum(dim=-1))  # sum over subject words
                upstream_node_values[key] = torch.stack(pattern, dim=0).mean(dim=0).cpu().item()
                        
    hook_func = partial(fetch_activations,
                        pre_word_positions=pre_verb_word_positions,
                        subj_positions=subj_ids)
    fwd_hooks = [(hook, hook_func) for hook in all_hooks]
    
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                return_type="loss",
                fwd_hooks=fwd_hooks
            )
    upstream_node_values = dict(sorted(upstream_node_values.items(), key=lambda x: x[1], reverse=True))
    avg_attn_at_subj = sum(upstream_node_values.values()) / len(upstream_node_values)
    
    sentence_tokens_num = [len(model.tokenizer.tokenize(sentence)) for sentence in text]
    avg_probs = [1 / num for num in sentence_tokens_num]
    avg_probs_overall = sum(avg_probs) / len(avg_probs)
    
    if sv_mode == "sv":
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/sv/pattern"
    else:
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/svd/pattern"
    if data_type == '0':
        save_path = os.path.join(save_dir, "subj_number_head_attn_at_subj_pick_0.jsonl")
    elif data_type == '1':
        save_path = os.path.join(save_dir, "subj_number_head_attn_at_subj_pick_1.jsonl")
    elif data_type == "mix":
        save_path = os.path.join(save_dir, "subj_number_head_attn_at_subj_pick_mix.jsonl")
    else:
        save_path = os.path.join(save_dir, "subj_number_head_attn_at_subj.jsonl")
    with jsonlines.open(os.path.join(save_dir, save_path), "w") as f:
        f.write({"avg_probs_over_all_tokens": avg_probs_overall})
        f.write({"avg_attn_at_subj": avg_attn_at_subj})
        for key, value in upstream_node_values.items():
            f.write({key: value})

    return upstream_node_values

def check_attention_max_at_subj(sv_mode="sv", data_type=None):
    """
    Check for all heads the attention pattern at the END token back to the subject.
    """
    
    head_subj_number_sv = ["head.8.5", "head.10.9"]
    head_subj_number_svd = ["head.6.0", "head.5.2", "head.4.3"]
    
    # if sv_mode == "sv":
    #     upstream_nodes = head_subj_number_sv
    # else:
    #     upstream_nodes = head_subj_number_svd
    
    upstream_nodes = []    
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            upstream_nodes.append(f"head.{i}.{j}")
    
    upstream_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=upstream_nodes, downstream_nodes=None, info=["pattern"])
    upstream_hooks = list(set(upstream_node_hook_map.values()))
    all_hooks = upstream_hooks
    print(all_hooks)
    
    if data_type == '0':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
    elif data_type == '1':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
    elif data_type == "mix":
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = lines[:] if data_type is not None else lines[:256]
    
    text = [item["clean_text"] for item in lines]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [item["clean_verb_pos"][0] - 1 for item in lines]
    subj_ids = [item["subj_pos"] for item in lines]
    
    upstream_node_values = {}
    
    def fetch_activations(activation, hook, pre_word_positions, subj_positions):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in upstream_node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                score = []
                for i in range(activation.shape[0]):
                    _, max_attn_pos = torch.max(activation[i, head_id, pre_word_positions[i], :], dim=-1)
                    if max_attn_pos in subj_positions[i]:
                        score.append(1)
                    else:
                        score.append(0)
                upstream_node_values[key] = sum(score) / len(score)
                        
    hook_func = partial(fetch_activations,
                        pre_word_positions=pre_verb_word_positions,
                        subj_positions=subj_ids)
    fwd_hooks = [(hook, hook_func) for hook in all_hooks]
    
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                return_type="loss",
                fwd_hooks=fwd_hooks
            )
    upstream_node_values = dict(sorted(upstream_node_values.items(), key=lambda x: x[1], reverse=True))
    
    save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/sv/pattern"
    if data_type == '0':
        save_path = os.path.join(save_dir, "subj_number_head_attn_max_at_subj_pick_0.jsonl")
    elif data_type == '1':
        save_path = os.path.join(save_dir, "subj_number_head_attn_max_at_subj_pick_1.jsonl")
    elif data_type == "mix":
        save_path = os.path.join(save_dir, "subj_number_head_attn_max_at_subj_pick_mix.jsonl")
    else:
        save_path = os.path.join(save_dir, "subj_number_head_attn_max_at_subj.jsonl")
    with jsonlines.open(os.path.join(save_dir, save_path), "w") as f:
        for key, value in upstream_node_values.items():
            f.write({key: value})

    return upstream_node_values

def patch_and_check_prod_variations(model, sv_mode="sv"):
    """
    We've already found some heads that conveys information contains the number of the subject.
    Now want to find out which heads affect those heads most.
    """
    head_subj_number_sv = ["head.8.5", "head.10.9"]
    head_subj_number_svd = ["head.6.0", "head.5.2", "head.4.3"]
    
    if sv_mode == "sv":
        nodes_to_check = head_subj_number_sv
    else:
        nodes_to_check = head_subj_number_svd
    
    upstream_nodes = []    
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            upstream_nodes.append(f"head.{i}.{j}")
    for node in nodes_to_check:
        upstream_nodes.remove(node)
    
    check_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=nodes_to_check, downstream_nodes=None)
    upstream_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=upstream_nodes, downstream_nodes=None)
    check_node_hooks = list(set(check_node_hook_map.values()))
    
    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_32.jsonl"
    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = lines[:256]
    
    text = [item["clean_text"] for item in lines]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    verb_ids = [item["clean_verb_ids"][0] for item in lines]
    anti_verb_ids = [item["corr_verb_ids"][0] for item in lines]
    pre_verb_word_positions = [item["clean_verb_pos"][0] - 1 for item in lines]
    
    logit_lens = []
    for i in range(len(text)):
        logit_diff = model.W_U[:, verb_ids[i]] - model.W_U[:, anti_verb_ids[i]]
        logit_lens.append(logit_diff)
    
    # def fetch_attn_pattern_activations(activation, hook, node_hook_map, node_vals, pre_word_positions, subj_positions):
        
    #     # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
    #     for key, val in node_hook_map.items():
    #         if val == hook.name:
    #             head_id = int(key.split(".")[2])
    #             pattern = []
    #             for i in range(activation.shape[0]):
    #                 pattern.append(activation[i, head_id, pre_word_positions[i], subj_positions[i]].sum(dim=-1))  # sum over subject words
    #             node_vals[key] = torch.stack(pattern, dim=0).mean(dim=0).cpu().item()  # average over batch
    
    def fetch_attn_result_activations(activation, hook, node_hook_map, node_vals, pre_verb_positions):
        
        # head.4.4 (blocks.4.attn.hook_result): [batch, len, head_index, d_model]
        for key, val in node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                acts = []
                for i in range(activation.shape[0]):
                    acts.append(activation[i, pre_verb_positions[i], head_id, :])  # (d_model)
                node_vals[key] = torch.stack(acts, dim=0)  # (bsz, d_model)
    
    # First, we collect the activations of the heads that convey the number of the subject before patching, and calculate their prods with logit lens.
    check_node_values_before_patching = {}               
    hook_func = partial(fetch_attn_result_activations,
                        node_hook_map=check_node_hook_map,
                        node_vals=check_node_values_before_patching,
                        pre_verb_positions=pre_verb_word_positions)
    fwd_hooks = [(hook, hook_func) for hook in check_node_hooks]
    
    model.reset_hooks()
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                fwd_hooks=fwd_hooks,
                return_type="loss"
            )
        
    head_logitlens_prods_before = {}
    with torch.no_grad():
        for key, value in check_node_values_before_patching.items():
            head_act_at_pre_word_after_ln = model.ln_final(value)  # (bsz, d_model)
            prods = []
            for i in range(len(pre_verb_word_positions)):
                head_act = head_act_at_pre_word_after_ln[i, :]  # (d_model)
                prod = torch.matmul(head_act, logit_lens[i].T)  # (d_model)*(d_model) -> (1)
                prods.append(prod)  
            head_logitlens_prods_before[key] = torch.stack(prods, dim=0).mean(dim=0).cpu()  # average over batch -> (1)
    
    
    # Next, patch the heads and check the variations of the heads that convey the number of the subject.
    
    def patch_head(activation, hook, head_idx, pre_verb_positions):
        
        # blocks.4.attn.hook_pattern (bsz, len, n_head, d_model)
        mean_act = activation.mean(dim=0).mean(dim=0).mean(dim=0)  # (d_model)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], head_idx, :] = mean_act
                    
    check_node_values_after_patching = {}
    # check_attn_diff = torch.zeros((len(check_hooks), len(upstream_hooks)))
    upstream_nodes_influence = {}
    # upstream_nodes_idx_map = {hook: int(hook.split('.')[1]) * model.cfg.n_heads + int(hook.split('.')[2]) for hook in upstream_nodes}  # head.4.4 -> 4*12+4
    with torch.no_grad():
        for node, hook_name in tqdm(upstream_node_hook_map.items()):
            head_idx = int(node.split(".")[2])  # head.4.4
            patch_hook = partial(patch_head,
                                 head_idx=head_idx,
                                 pre_verb_positions=pre_verb_word_positions
                                 )
            check_hook_func = partial(fetch_attn_result_activations,
                        node_hook_map=check_node_hook_map,
                        node_vals=check_node_values_after_patching,
                        pre_verb_positions=pre_verb_word_positions
                        )
            check_hooks = [(hook, check_hook_func) for hook in check_node_hooks]
            fwd_hooks = check_hooks + [(hook_name, patch_hook)]
            
            model.reset_hooks()
            _ = model.run_with_hooks(
                    input=text_inputs["input_ids"].to(model.cfg.device),
                    attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                    return_type="loss",
                    fwd_hooks=fwd_hooks
                )
            
            head_logitlens_prods_after = {}
            with torch.no_grad():
                for key, value in check_node_values_after_patching.items():
                    head_act_at_pre_word_after_ln = model.ln_final(value)  # (bsz, d_model)
                    prods = []
                    for i in range(len(pre_verb_word_positions)):
                        head_act = head_act_at_pre_word_after_ln[i, :]
                        prod = torch.matmul(head_act, logit_lens[i].T)
                        prods.append(prod)
                    head_logitlens_prods_after[key] = torch.stack(prods, dim=0).mean(dim=0).cpu()  # (1)
            
            # influence is the variance in the prods averaged over the check_nodes(subj_number heads)
            prod_before = torch.stack([prod for prod in head_logitlens_prods_before.values()], dim=0)
            prod_after = torch.stack([prod for prod in head_logitlens_prods_after.values()], dim=0)  # (n_check_nodes, 1)
            prods_diff = prod_after - prod_before
            upstream_nodes_influence[node] = prods_diff.mean(dim=0).cpu().item()  # (1)
            
    for node in nodes_to_check:
        upstream_nodes_influence[node] = 0
    
    # draw graph
    data = np.array([[upstream_nodes_influence[f"head.{i}.{j}"]  for j in range(model.cfg.n_heads)] for i in range(model.cfg.n_layers)])
    plt.figure(figsize=(10, 8))
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vmax = data.max(), vcenter=0)
    plt.imshow(data, cmap=plt.cm.PiYG, aspect='auto', norm=norm)  # PiYG(light green, violet)
    plt.colorbar()

    plt.xlabel('Attention Head')
    plt.ylabel('Layer')

    plt.xticks(np.arange(model.cfg.n_heads))
    plt.yticks(np.arange(model.cfg.n_layers))

    task = "subject-verb agreement" if sv_mode == "sv" else "subject-verb disagreement"
    plt.title('Patching influence on Subject Number Heads from other heads\nin the {} task'.format(task))
    if sv_mode == "sv":
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/sv"
    else:
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/svd"
    save_path = os.path.join(save_dir, f"patching_influence_prod.pdf")
    plt.savefig(save_path)

def patch_and_check_attn_variations(model, sv_mode="sv", data_type=None):
    """
    We've already found some heads that conveys information contains the number of the subject.
    Now want to find out which heads affect those heads most.
    """
    head_subj_number_sv = ["head.8.5", "head.10.9"]
    head_subj_number_sv = ["head.10.9"]
    head_subj_number_sv = ["head.8.5"]
    head_subj_number_svd = ["head.6.0", "head.5.2", "head.4.3", "head.8.5", "head.10.9"]
    head_subj_number_svd = ["head.10.9"]
    
    if sv_mode == "sv":
        nodes_to_check = head_subj_number_sv
    else:
        nodes_to_check = head_subj_number_svd
    
    upstream_nodes = []
    layer_max = int(nodes_to_check[0].split(".")[1])
    head_max = int(nodes_to_check[0].split(".")[2])
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            if i < layer_max or (i == layer_max and j < head_max): 
                upstream_nodes.append(f"head.{i}.{j}")
    print(upstream_nodes)
    # for node in nodes_to_check:
    #     upstream_nodes.remove(node)
    
    check_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=nodes_to_check, downstream_nodes=None, info=["pattern"])
    upstream_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=upstream_nodes, downstream_nodes=None)
    check_node_hooks = list(set(check_node_hook_map.values()))
    
    if data_type == '0':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type0.jsonl"
    elif data_type == '1':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type1.jsonl"
    elif data_type == "mix":
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = lines[:] if len(lines) < 126 else lines[:128]
    
    text = [item["clean_text"] for item in lines]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [item["clean_verb_pos"][0] - 1 for item in lines]
    subj_pos = [item["subj_pos"] for item in lines]
    
    
    def fetch_attn_pattern_activations(activation, hook, node_hook_map, node_vals, pre_word_positions, subj_positions):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                pattern = []
                for i in range(activation.shape[0]):
                    pattern.append(activation[i, head_id, pre_word_positions[i], subj_positions[i]].sum(dim=-1))  # sum over subject words -> (1)
                node_vals[key] = torch.stack(pattern, dim=0).mean(dim=0).cpu().item()  # average over batch
    
    def fetch_attn_pattern_max_score_activations(activation, hook, node_hook_map, node_vals, pre_word_positions, subj_positions):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                score = []
                for i in range(activation.shape[0]):
                    _, max_attn_pos = torch.max(activation[i, head_id, pre_word_positions[i], :], dim=-1)
                    if max_attn_pos in subj_positions[i]:
                        score.append(1)
                    else:
                        score.append(0)
                node_vals[key] = sum(score) / len(score)
                
    
    # First, we collect the activations of the heads that convey the number of the subject before patching, and calculate their prods with logit lens.
    head_pattern_before = {}               
    hook_func = partial(fetch_attn_pattern_activations,
                        node_hook_map=check_node_hook_map,
                        node_vals=head_pattern_before,
                        pre_word_positions=pre_verb_word_positions,
                        subj_positions=subj_pos)
    fwd_hooks = [(hook, hook_func) for hook in check_node_hooks]
    
    model.reset_hooks()
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                fwd_hooks=fwd_hooks,
                return_type="loss"
            )
    
    # Next, patch the heads and check the variations of the heads that convey the number of the subject.
    
    def patch_head(activation, hook, head_idx, pre_verb_positions):
        
        # blocks.4.attn.hook_pattern (bsz, len, n_head, d_model)
        mean_act = activation.mean(dim=0).mean(dim=0).mean(dim=0) # (d_model)
        mean_act = activation[:, :, head_idx, :].mean(dim=0).mean(dim=0) # (d_model)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], head_idx, :] = mean_act
                    
    head_pattern_after = {}
    upstream_nodes_influence = {}
    with torch.no_grad():
        for node, hook_name in tqdm(upstream_node_hook_map.items()):
            head_idx = int(node.split(".")[2])  # head.4.4
            patch_hook = partial(patch_head,
                                 head_idx=head_idx,
                                 pre_verb_positions=pre_verb_word_positions
                                 )
            check_hook_func = partial(fetch_attn_pattern_activations,
                        node_hook_map=check_node_hook_map,
                        node_vals=head_pattern_after,
                        pre_word_positions=pre_verb_word_positions,
                        subj_positions=subj_pos
                        )
            check_hooks = [(hook, check_hook_func) for hook in check_node_hooks]
            fwd_hooks = check_hooks + [(hook_name, patch_hook)]
            
            model.reset_hooks()
            _ = model.run_with_hooks(
                    input=text_inputs["input_ids"].to(model.cfg.device),
                    attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                    return_type="loss",
                    fwd_hooks=fwd_hooks
                )
            
            # influence is the variance in the prods averaged over the check_nodes(subj_number heads)
            pattern_before = np.array([val for val in head_pattern_before.values()])
            pattern_after = np.array([val for val in head_pattern_after.values()])
            pattern_diff = np.mean(pattern_after - pattern_before)
            print(pattern_before, pattern_after, pattern_diff)
            upstream_nodes_influence[node] = pattern_diff
            
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            if f"head.{i}.{j}" not in upstream_nodes_influence:
                upstream_nodes_influence[f"head.{i}.{j}"] = 0
            else:
                if upstream_nodes_influence[f"head.{i}.{j}"] > 0:
                    upstream_nodes_influence[f"head.{i}.{j}"] = 0
                else:
                    upstream_nodes_influence[f"head.{i}.{j}"] = -upstream_nodes_influence[f"head.{i}.{j}"]
    
    # draw graph
    data = np.array([[upstream_nodes_influence[f"head.{i}.{j}"] for j in range(model.cfg.n_heads)] for i in range(model.cfg.n_layers)])
    plt.figure(figsize=(10, 8))
    # norm = mcolors.TwoSlopeNorm(vmin=data.min(), vmax = data.max(), vcenter=0)
    plt.imshow(data, cmap=plt.cm.Greens, aspect='auto')  # PiYG(light green, violet)
    plt.colorbar()

    plt.xlabel('Attention Head')
    plt.ylabel('Layer')

    plt.xticks(np.arange(model.cfg.n_heads))
    plt.yticks(np.arange(model.cfg.n_layers))

    task = "subject-verb agreement" if sv_mode == "sv" else "subject-verb disagreement"
    plt.title('Patching influence on Subject Number Heads from other heads\nin the {} task'.format(task))
    if sv_mode == "sv":
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/sv"
    else:
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/sv_analyses/head_subj_number/svd"
    
    if data_type == '0':
        save_path = os.path.join(save_dir, "patching_influence_pick_0.pdf")
    elif data_type == '1':
        save_path = os.path.join(save_dir, "patching_influence_pick_1.pdf")
    elif data_type == "mix":
        save_path = os.path.join(save_dir, "patching_influence_pick_mix.pdf")
    else:
        save_path = os.path.join(save_dir, "patching_influence.pdf")
    save_path = save_path.split(".")[0] + f"_{str(layer_max)}-{str(head_max)}.pdf"
    plt.savefig(save_path)

def patch_and_check_attn_variations_together(model, sv_mode="sv", data_type=None):
    """
    We've already found some heads that conveys information contains the number of the subject.
    Now want to find out which heads affect those heads most.
    This function is to check the influence on a certain head from patching a group of heads.
    """
    head_subj_number_sv = ["head.8.5", "head.10.9"]
    head_subj_number_sv = ["head.10.9"]
    head_subj_number_sv = ["head.8.5"]
    head_subj_number_svd = ["head.6.0", "head.5.2", "head.4.3"]
    
    if sv_mode == "sv":
        nodes_to_check = head_subj_number_sv
    else:
        nodes_to_check = head_subj_number_svd
    
    upstream_nodes = ["head.2.11", "head.7.4"]
    print(upstream_nodes)
    # for node in nodes_to_check:
    #     upstream_nodes.remove(node)
    
    check_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=nodes_to_check, downstream_nodes=None, info=["pattern"])
    upstream_node_hook_map, _ = get_hooks_from_nodes(upstream_nodes=upstream_nodes, downstream_nodes=None)
    check_node_hooks = list(set(check_node_hook_map.values()))
    
    if data_type == '0':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
    elif data_type == '1':
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type1.jsonl"
    elif data_type == "mix":
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = lines[:] if len(lines) < 126 else lines[:128]
    
    text = [item["clean_text"] for item in lines]
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    pre_verb_word_positions = [item["clean_verb_pos"][0] - 1 for item in lines]
    subj_pos = [item["subj_pos"] for item in lines]
    
    
    def fetch_attn_pattern_activations(activation, hook, node_hook_map, node_vals, pre_word_positions, subj_positions):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                pattern = []
                for i in range(activation.shape[0]):
                    pattern.append(activation[i, head_id, pre_word_positions[i], subj_positions[i]].sum(dim=-1))  # sum over subject words -> (1)
                node_vals[key] = torch.stack(pattern, dim=0).mean(dim=0).cpu().item()  # average over batch
    
    def fetch_attn_pattern_max_score_activations(activation, hook, node_hook_map, node_vals, pre_word_positions, subj_positions):
        
        # head.4.4 (blocks.4.attn.hook_pattern): [batch, head_index, query_pos, key_pos]
        for key, val in node_hook_map.items():
            if val == hook.name:
                head_id = int(key.split(".")[2])
                score = []
                for i in range(activation.shape[0]):
                    _, max_attn_pos = torch.max(activation[i, head_id, pre_word_positions[i], :], dim=-1)
                    if max_attn_pos in subj_positions[i]:
                        score.append(1)
                    else:
                        score.append(0)
                node_vals[key] = sum(score) / len(score)
                
    
    # First, we collect the activations of the heads that convey the number of the subject before patching, and calculate their prods with logit lens.
    head_pattern_before = {}               
    hook_func = partial(fetch_attn_pattern_activations,
                        node_hook_map=check_node_hook_map,
                        node_vals=head_pattern_before,
                        pre_word_positions=pre_verb_word_positions,
                        subj_positions=subj_pos)
    fwd_hooks = [(hook, hook_func) for hook in check_node_hooks]
    
    model.reset_hooks()
    with torch.no_grad():
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                fwd_hooks=fwd_hooks,
                return_type="loss"
            )
    
    # Next, patch the heads and check the variations of the heads that convey the number of the subject.
    
    def patch_head(activation, hook, head_idx, pre_verb_positions):
        
        # blocks.4.attn.hook_pattern (bsz, len, n_head, d_model)
        # mean_act = activation.mean(dim=0).mean(dim=0).mean(dim=0) # (d_model)
        mean_act = activation[:, :, head_idx, :].mean(dim=0).mean(dim=0) # (d_model)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], head_idx, :] = mean_act
                    
    head_pattern_after = {}
    upstream_nodes_influence = {}
    with torch.no_grad():
        patch_hooks = []
        for node, hook_name in tqdm(upstream_node_hook_map.items()):
            head_idx = int(node.split(".")[2])
            patch_hook = partial(patch_head,
                                head_idx=head_idx,
                                pre_verb_positions=pre_verb_word_positions
                                )
            patch_hooks.append((hook_name, patch_hook))
        check_hook_func = partial(fetch_attn_pattern_activations,
                    node_hook_map=check_node_hook_map,
                    node_vals=head_pattern_after,
                    pre_word_positions=pre_verb_word_positions,
                    subj_positions=subj_pos
                    )
        check_hooks = [(hook, check_hook_func) for hook in check_node_hooks]
        fwd_hooks = check_hooks + patch_hooks
        
        model.reset_hooks()
        _ = model.run_with_hooks(
                input=text_inputs["input_ids"].to(model.cfg.device),
                attention_mask=text_inputs["attention_mask"].to(model.cfg.device),
                return_type="loss",
                fwd_hooks=fwd_hooks
            )
        
        # influence is the variance in the prods averaged over the check_nodes(subj_number heads)
        pattern_before = np.array([val for val in head_pattern_before.values()])
        pattern_after = np.array([val for val in head_pattern_after.values()])
        pattern_diff = np.mean(pattern_after - pattern_before)
        print(pattern_before, pattern_after, pattern_diff)
        upstream_nodes_influence[node] = pattern_diff

def calculate_faithfulness(model, log_path, mode):
    log = []
    with jsonlines.open(log_path) as f:
        for line in f:
            log.append(line)
    edges = log[0]["edges"]
    
    up_node_set = set()
    down_node_set = set()
    for up, down, _ in edges:
        up_node_set.add(up)
        down_node_set.add(down)
    print(len(up_node_set), len(down_node_set))
    
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    
    upstream_nodes=["mlp", "head"]
    downstream_nodes=["mlp", "head"]
    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
    
    up_node_to_ablate = [node for node in graph.upstream_nodes if node not in up_node_set]
    down_node_to_ablate = [node for node in graph.downstream_nodes if node not in down_node_set]
    print(len(up_node_to_ablate), len(down_node_to_ablate))
    
    # prepare data
    dev_dataset = SVDataset(model.tokenizer, "test")
    sv_collate_fn = SVCollateFn(model.tokenizer)
    eval_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=sv_collate_fn)
    
    # ablation
    def patch_head_0(activation, hook, head_idx, pre_verb_positions): 
        # blocks.4.hook_result/hook_q_input (bsz, len, n_head, d_model)
        mean_act = activation[:, :, head_idx, :].mean(dim=0).mean(dim=0) # (d_model)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], head_idx, :] = mean_act
    
    def patch_mlp_0(activation, hook, pre_verb_positions):
        # blocks.4.mlp_in/out (bsz, len, d_model)
        mean_act = activation.mean(dim=0).mean(dim=0)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], :] = mean_act
            
    def patch_head(activation, hook, head_idx, pre_verb_positions): 
        # blocks.4.hook_result/hook_q_input (bsz, len, n_head, d_model)
        for i in range(activation.shape[0]):
            mean_act = activation[i, pre_verb_positions[i], head_idx, :].mean(dim=0).mean(dim=0) # (d_model)
            activation[i, pre_verb_positions[i], head_idx, :] = mean_act
    
    def patch_mlp(activation, hook, pre_verb_positions):
        # blocks.4.mlp_in/out (bsz, len, d_model)
        for i in range(activation.shape[0]):
            mean_act = activation[i, pre_verb_positions[i], :].mean(dim=0).mean(dim=0)
            activation[i, pre_verb_positions[i], :] = mean_act
    
    upstream_node_hook_map, downstream_node_hook_map = get_hooks_from_nodes(upstream_nodes=up_node_to_ablate, downstream_nodes=down_node_to_ablate)
    ablate_node_hook_map = {**upstream_node_hook_map, **downstream_node_hook_map}
    
    # if empty circuit
    if mode == "empty":
        upstream_node_hook_map, downstream_node_hook_map = get_hooks_from_nodes(upstream_nodes=graph.upstream_nodes, downstream_nodes=graph.downstream_nodes)
        ablate_node_hook_map = {**upstream_node_hook_map, **downstream_node_hook_map}
    if mode == "other":
        upstream_node_hook_map, downstream_node_hook_map = get_hooks_from_nodes(upstream_nodes=up_node_set, downstream_nodes=down_node_set)
        ablate_node_hook_map = {**upstream_node_hook_map, **downstream_node_hook_map}
    
    # forward
    model.reset_hooks()
    
    logit_diffs = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["clean_inputs"]["input_ids"].to(model.cfg.device)
            attention_mask = batch["clean_inputs"]["attention_mask"].to(model.cfg.device)
            pre_verb_word_positions = [[pos - 1 for pos in data] for data in batch["clean_verb_pos"]] 
            fwd_hooks = []
            for node_name, hook_name in ablate_node_hook_map.items():
                if "head" in node_name:
                    head_idx = int(node_name.split(".")[2])
                    patch_hook = partial(patch_head_0,
                                        head_idx=head_idx,
                                        pre_verb_positions=pre_verb_word_positions
                                        )
                else:
                    patch_hook = partial(patch_mlp_0,
                                        pre_verb_positions=pre_verb_word_positions
                                        )
                fwd_hooks.append((hook_name, patch_hook))
            if mode == "full":
                fwd_hooks = []
            
            logits = model.run_with_hooks(
                    input=input_ids,
                    attention_mask=attention_mask,
                    return_type="logits",
                    fwd_hooks=fwd_hooks
                )

            # logit difference
            logit_diff = avg_logit_diff_sv(logits, batch, per_prompt=False)
            logit_diffs.append(logit_diff.item())
    
    logit_diff_final = sum(logit_diffs) / len(logit_diffs)
    print(abs(logit_diff_final))
    

if __name__ == "__main__":
    model_name = "gpt2-small"
    ckpt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/model-steps_1499_epoch_3.pt"
    # ckpt_path = None
    device = "cuda:1"
    model = load_model(model_name, ckpt_path, device)
    sv_mode = "sv" if not ckpt_path else "svd"
    get_heads_for_logit_lens(sv_mode=sv_mode)  # *
    # check_single_attention_pattern(model, sv_mode=sv_mode, data_type="1")
    # check_attention_at_subj(sv_mode=sv_mode, data_type='1')  # *
    # check_attention_max_at_subj(sv_mode=sv_mode, data_type="mix")
    # patch_and_check_prod_variations(model, sv_mode=sv_mode)
    # patch_and_check_attn_variations(model, sv_mode=sv_mode, data_type=None)  # *
    # patch_and_check_attn_variations_together(model, sv_mode=sv_mode, data_type='1')

    