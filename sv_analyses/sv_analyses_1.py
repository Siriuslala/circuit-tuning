"""
Script for analyzing the subject-verb agreement task.
"""

from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import torch

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(str(root_dir))

from utils import load_model_from_ckpt
from utils import load_model, load_model_from_ckpt, load_model_new, hookdedTF_to_TF

from eap.eap_graph_old import EAPGraph
from eap.eap_wrapper import EAP_ablation
from circuit_data import SVDataset, SVCollateFn, BiasDataset, BiasCollateFn
from eap.patching_metrics import (
    avg_logit_diff_sv, 
    avg_neg_log_prob_diff_sv,
    avg_logit_diff_bias,
    patching_metric
)

import numpy as np
import dataclasses
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import jsonlines
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from collections import defaultdict
from sklearn.decomposition import PCA

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

def load_model_old(model_name, ckpt_path, device, split_params=False):
    if "gpt" in model_name.lower():
        hf_model_name = "gpt2"
    else:
        hf_model_name = model_name
    cfg_path = hf_hub_download(hf_model_name, "config.json", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, local_files_only=True)
    if ckpt_path:
        model = load_model_from_ckpt(
            model_name,
            ckpt_path,
            cfg_path,
            split_params=split_params,
            tokenizer=tokenizer
        )
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
    
def get_heads_for_logit_lens(model, sv_mode="sv", batch_size=32):
    
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
    data_path = root_dir / "data/sv_dataset/test_3000.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = [item for item in lines if len(item["clean_verb_pos"]) == 1]    
    
    sample_num = batch_size
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
    data = np.array([[head_logitlens_prods[f"head.{i}.{j}"] for j in range(model.cfg.n_heads)] for i in range(model.cfg.n_layers)])
    plt.figure(figsize=(10, 8))

    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vmax = data.max(), vcenter=0)
    # plt.imshow(data, cmap=plt.cm.PiYG, aspect='auto', norm=norm)  # PiYG(light green, violet) 发散调色板

    coral_to_white = ["#F0C20A", '#FFFFFF']  # 珊瑚色到白色
    white_to_gray = ['#FFFFFF', '#808080']  # 白色到灰色

    cmap1 = LinearSegmentedColormap.from_list('coral_to_white', coral_to_white, N=128)
    cmap2 = LinearSegmentedColormap.from_list('white_to_gray', white_to_gray, N=128)

    colors = np.vstack((cmap1(np.linspace(0, 1, 128)), cmap2(np.linspace(0, 1, 128))))
    two_slope_cmap = LinearSegmentedColormap.from_list('two_slope_gray_white_coral', colors)

    plt.imshow(data, cmap=two_slope_cmap, aspect='auto', norm=norm)
    # plt.imshow(data, cmap=two_slope_cmap, aspect='auto', vmin=data.min(), vmax=data.max())
    plt.colorbar()
    # set the fontsize of the colorbar ticks
    cbar = plt.gcf().axes[-1]
    cbar.yaxis.set_tick_params(labelsize=18)

    plt.xlabel('Attention Head', fontsize=25)
    plt.ylabel('Layer', fontsize=25)

    plt.xticks(np.arange(model.cfg.n_heads), fontsize=18)
    plt.yticks(np.arange(model.cfg.n_layers), fontsize=18)
    
    plt.tight_layout()

    task = "subject-verb agreement" if sv_mode == "sv" else "subject-verb disagreement"
    # plt.title('Inner product between attention head ouputs and logit difference\nin the {} task'.format(task))
    if sv_mode == "sv":
        save_dir = root_dir / "figures/flip_heads/head_subj_number/sv"
    else:
        save_dir = root_dir / "figures/flip_heads/head_subj_number/svd"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"head_logitlens_prods_sample{sample_num}.pdf"
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
    save_dir = str(root_dir / "sv_analyses/head_subj_number/sv/pattern")
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

def check_attention_at_subj(model, sv_mode="sv", data_type=None):
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
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
    elif data_type == '1':
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
    elif data_type == "mix":
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
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
        save_dir = str(root_dir / "sv_analyses/head_subj_number/sv/pattern")
    else:
        save_dir = str(root_dir / "sv_analyses/head_subj_number/svd/pattern")
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

def check_attention_max_at_subj(model,sv_mode="sv", data_type=None):
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
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
    elif data_type == '1':
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
    elif data_type == "mix":
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
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
    
    save_dir = str(root_dir / "sv_analyses/head_subj_number/sv/pattern")
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
    
    data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_32.jsonl"
    data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
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
        save_dir = str(root_dir / "sv_analyses/head_subj_number/sv")
    else:
        save_dir = str(root_dir / "sv_analyses/head_subj_number/svd")
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
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type0.jsonl"
    elif data_type == '1':
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type1.jsonl"
    elif data_type == "mix":
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
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
        save_dir = str(root_dir / "sv_analyses/head_subj_number/sv")
    else:
        save_dir = str(root_dir / "sv_analyses/head_subj_number/svd")
    
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
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_0.jsonl"
    elif data_type == '1':
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_20_1.jsonl"
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type1.jsonl"
    elif data_type == "mix":
        data_path = root_dir / "data/sv_dataset/test_3000_subj_pick_40_mix.jsonl"
    else:
        data_path = root_dir / "data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
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

def get_circuit_for_a_batch(batch, model, graph, topn=1000, return_total_graph=False):
    # reset graph
    graph.reset_scores()

    # get config
    @dataclass
    class Config:
        task: str = "sv"
        top_n: int = topn
        ie_over_seq: bool = False
        process_or_outcome: str = "outcome"
        ablation_method: str = "mean"
        smooth: bool = False
        threshold: float = 0.0
        cross_layer: bool = True
        prune_method: str = "top_edges"

    config = Config()

    # get a new subgraph
    eap_func = EAP_ablation
    metric_func = avg_logit_diff_sv
    patching_metric_fn = partial(patching_metric, logit_diff_func=metric_func)
    graph, _, _ = eap_func(
        model,
        graph=graph,
        batch=batch,
        metric=patching_metric_fn,
        config=config,
        )
    total_edge_num = len(graph.upstream_nodes) * len(graph.downstream_nodes)
    edge_num = total_edge_num if return_total_graph else topn
    top_edges = graph.top_edges(
        n=edge_num, 
        abs_scores=True, 
        cross_layer=config.cross_layer, 
        prune_method=config.prune_method
    )
    return top_edges

def get_sv_circuits(model_name, ckpt_path=None, device="cuda", task="svd", topn=1000, batch_size=32, return_total_graph=False):

    # Load model
    model, tokenizer = load_model_new(model_name, ckpt_path, device, use_transformer_lens=True)
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)

    upstream_node_type=["mlp", "head"]
    downstream_node_type=["mlp", "head"]
    graph = EAPGraph(model.cfg, upstream_node_type, downstream_node_type)

    # Load dataset
    dataset = SVDataset(model.tokenizer, "dev")
    sv_collate_fn = SVCollateFn(model.tokenizer, task=task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=sv_collate_fn)
    
    # Find circuits
    for idx, batch in enumerate(dataloader):
        top_edges = get_circuit_for_a_batch(batch, model, graph, topn=topn, return_total_graph=return_total_graph)
        break
    return top_edges

def get_ablate_node_hook_map(graph, mode, circuit_edges):
    up_node_set = set()
    down_node_set = set()
    for up, down, _ in circuit_edges:
        up_node_set.add(up)
        down_node_set.add(down)
        
    if mode == "full":
        up_node_to_ablate = []
        down_node_to_ablate = []
    elif mode == "empty":
        up_node_to_ablate = graph.upstream_nodes
        down_node_to_ablate = graph.downstream_nodes
    elif mode == "circuit":
        up_node_to_ablate = [node for node in graph.upstream_nodes if node not in up_node_set]
        down_node_to_ablate = [node for node in graph.downstream_nodes if node not in down_node_set]
    elif mode == "other":
        up_node_to_ablate = list(up_node_set)
        down_node_to_ablate = list(down_node_set)
    else:
        raise ValueError("mode must be one of 'full', 'empty', 'circuit', 'other'")
    
    upstream_node_hook_map, downstream_node_hook_map = get_hooks_from_nodes(upstream_nodes=up_node_to_ablate, downstream_nodes=down_node_to_ablate)
    ablate_node_hook_map = {**upstream_node_hook_map, **downstream_node_hook_map}

    return ablate_node_hook_map

def sub_calculate_faithfulness_and_completeness(
    model,
    mode,
    log_path=None,
    mean_ablation=True,
    batch_size=4,
    task="svd",
    topn=1000,
):
    """ 
    mode: "full", "empty", "circuit", "other"
    log_path: if not None, use the last circuit in the log_path

    """
    
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    
    upstream_nodes=["mlp", "head"]
    downstream_nodes=["mlp", "head"]
    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)

    ablate_node_hook_map = None
    if log_path is not None:
        log_edges = []
        with jsonlines.open(log_path) as f:
            for line in f:
                keys = ["edges", "edge_info"]
                for key in keys:
                    if key in line:
                        log_edges.append(line[key])
        edges = log_edges[-1]
        ablate_node_hook_map = get_ablate_node_hook_map(graph, mode, edges)
    
    # prepare data
    if mean_ablation:
        dev_dataset = SVDataset(model.tokenizer, "dev")
        sv_collate_fn = SVCollateFn(model.tokenizer, task=task)
    else:
        dev_dataset = SVDataset(model.tokenizer, "dev", use_contrastive=True)
        sv_collate_fn = SVCollateFn(model.tokenizer, clean_corrupted_together=True, task=task)
    eval_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=sv_collate_fn)
    
    def patch_head(activation, hook, head_idx, clean_pre_verb_positions, corr_pre_verb_positions): 
        # blocks.4.hook_result/hook_q_input (bsz, len, n_head, d_model)
        sample_num = activation.shape[0] // 2
        for i in range(sample_num):
            corr_act = activation[i * 2 + 1, corr_pre_verb_positions[i], head_idx, :]
            activation[i * 2, clean_pre_verb_positions[i], head_idx, :] = corr_act

    def patch_mlp(activation, hook, clean_pre_verb_positions, corr_pre_verb_positions):
        # blocks.4.mlp_in/out (bsz, len, d_model)
        sample_num = activation.shape[0] // 2
        for i in range(sample_num):
            corr_act = activation[i * 2 + 1, corr_pre_verb_positions[i], :]
            activation[i * 2, clean_pre_verb_positions[i], :] = corr_act
    
    def patch_head_ablation(activation, hook, head_idx, pre_verb_positions): 
        # blocks.4.hook_result/hook_q_input (bsz, len, n_head, d_model)
        mean_act = activation[:, :, head_idx, :].mean(dim=0).mean(dim=0) # (d_model)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], head_idx, :] = mean_act
    
    def patch_mlp_ablation(activation, hook, pre_verb_positions):
        # blocks.4.mlp_in/out (bsz, len, d_model)
        mean_act = activation.mean(dim=0).mean(dim=0)
        for i in range(activation.shape[0]):
            activation[i, pre_verb_positions[i], :] = mean_act

    # forward
    model.reset_hooks()
    logit_diffs = []
    for idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        if mean_ablation:
            input_ids = batch["clean_inputs"]["input_ids"].to(model.cfg.device)
            attention_mask = batch["clean_inputs"]["attention_mask"].to(model.cfg.device)
        else:
            input_ids = batch["all_inputs"]["input_ids"].to(model.cfg.device)
            attention_mask = batch["all_inputs"]["attention_mask"].to(model.cfg.device)
        clean_pre_verb_word_positions = [[pos - 1 for pos in data] for data in batch["clean_verb_pos"]]
        corr_pre_verb_word_positions = [[pos - 1 for pos in data] for data in batch["corr_verb_pos"]]

        # prepare nodes to ablate
        if log_path is None:
            if mean_ablation:
                edges = get_circuit_for_a_batch(batch, model, graph, topn=topn)
            else:
                clean_input_ids = batch["all_inputs"]["input_ids"][::2]
                clean_attention_mask = batch["all_inputs"]["attention_mask"][::2]
                corr_input_ids = batch["all_inputs"]["input_ids"][1::2]
                corr_attention_mask = batch["all_inputs"]["attention_mask"][1::2]
                new_batch = {
                    "clean_inputs": {"input_ids": clean_input_ids, "attention_mask": clean_attention_mask},
                    "corr_inputs": {"input_ids": corr_input_ids, "attention_mask": corr_attention_mask},
                    "clean_verb_pos": batch["clean_verb_pos"],
                    "corr_verb_pos": batch["corr_verb_pos"],
                    "clean_verb_ids": batch["clean_verb_ids"],
                    "corr_verb_ids": batch["corr_verb_ids"],
                }
                edges = get_circuit_for_a_batch(new_batch, model, graph)
            ablate_node_hook_map = get_ablate_node_hook_map(graph, mode, edges)
        
        # prepare inputs
        # if not mean_ablation:
        #     input_ids = batch["all_inputs"]["input_ids"][::2]
        #     attention_mask = batch["all_inputs"]["attention_mask"][::2]
        
        # start forward
        fwd_hooks = []
        for node_name, hook_name in ablate_node_hook_map.items():
            if "head" in node_name:
                head_idx = int(node_name.split(".")[2])
                if mean_ablation:
                    patch_hook = partial(
                        patch_head_ablation,
                        head_idx=head_idx,
                        pre_verb_positions=clean_pre_verb_word_positions,
                    )
                else:
                    patch_hook = partial(
                        patch_head,
                        head_idx=head_idx,
                        clean_pre_verb_positions=clean_pre_verb_word_positions,
                        corr_pre_verb_positions=corr_pre_verb_word_positions
                    )
            else:
                if mean_ablation:
                    patch_hook = partial(
                        patch_mlp_ablation,
                        pre_verb_positions=clean_pre_verb_word_positions,
                    )
                else:
                    patch_hook = partial(
                        patch_mlp,
                        clean_pre_verb_positions=clean_pre_verb_word_positions,
                        corr_pre_verb_positions=corr_pre_verb_word_positions
                    )
            fwd_hooks.append((hook_name, patch_hook))

        with torch.no_grad():
            logits = model.run_with_hooks(
                input=input_ids,
                attention_mask=attention_mask,
                return_type="logits",
                fwd_hooks=fwd_hooks
            )
        model.reset_hooks()

        # logit difference
        if not mean_ablation:
            logits = logits[::2]  # only keep clean samples
        logit_diff = avg_logit_diff_sv(logits, batch, per_prompt=True)
        logit_diffs.extend(logit_diff)
        if idx + 1 >= 2:  # evaluate on 50 samples
            break
    
    # logit_diff_final = sum(logit_diffs) / len(logit_diffs)
    # print(f"mode: {mode}, logit_diff_final: {logit_diff_final}")
    return logit_diffs

def find_outliers_mad(data, threshold=3.5):
    """
    使用 MAD (中位数绝对偏差) 检测离群点
    """
    data = np.array(data)

    median = np.median(data)
    abs_deviation = np.abs(data - median)
    mad = np.median(abs_deviation)
    
    # 计算修正后的 Z-Score (Modified Z-Score)
    # 0.6745 是正态分布的标准缩放因子
    if mad == 0: # 防止除以0
        return []
    modified_z_score = 0.6745 * abs_deviation / mad
    
    outlier_indices = np.where(modified_z_score > threshold)[0]
    return outlier_indices, data[outlier_indices]

def calculate_faithfulness_and_completeness(
    model,
    log_path,
    mean_ablation=True,
    batch_size=4,
    task="svd",
    topn=1000,
):
    """
    single: if True, use the act from the contrastive prompt for patching
    """
    logit_diff_full = sub_calculate_faithfulness_and_completeness(model, mode="full", log_path=log_path, mean_ablation=mean_ablation, batch_size=batch_size, task=task, topn=topn)
    logit_diff_empty = sub_calculate_faithfulness_and_completeness(model, mode="empty", log_path=log_path, mean_ablation=mean_ablation, batch_size=batch_size, task=task, topn=topn)
    logit_diff_circuit = sub_calculate_faithfulness_and_completeness(model, mode="circuit", log_path=log_path, mean_ablation=mean_ablation, batch_size=batch_size, task=task, topn=topn)
    logit_diff_other = sub_calculate_faithfulness_and_completeness(model, mode="other", log_path=log_path, mean_ablation=mean_ablation, batch_size=batch_size, task=task, topn=topn)
    # print(f"logit_diff_full: {logit_diff_full}")
    # print(f"logit_diff_empty: {logit_diff_empty}")
    # print(f"logit_diff_circuit: {logit_diff_circuit}")
    # print(f"logit_diff_other: {logit_diff_other}")
    
    faithfulness = []
    completeness = []
    for i in range(len(logit_diff_full)):
        faithfulness.append((logit_diff_circuit[i] - logit_diff_empty[i]) / (logit_diff_full[i] - logit_diff_empty[i]))
        completeness.append((logit_diff_other[i] - logit_diff_empty[i]) / (logit_diff_full[i] - logit_diff_empty[i]))
    
    faithfulness = np.array(faithfulness)
    completeness = np.array(completeness)

    # filter outliers
    outlier_indices, outlier_values = find_outliers_mad(faithfulness, threshold=2.0)
    if len(outlier_indices) > 0:
        print(f"Outliers in faithfulness at indices {outlier_indices}: {outlier_values}")
        faithfulness = [val for idx, val in enumerate(faithfulness) if idx not in outlier_indices]

    outlier_indices, outlier_values = find_outliers_mad(completeness, threshold=2.0)
    if len(outlier_indices) > 0:
        print(f"Outliers in completeness at indices {outlier_indices}: {outlier_values}")
        completeness = [val for idx, val in enumerate(completeness) if idx not in outlier_indices]
    
    avg_faithfulness = sum(faithfulness) / len(faithfulness)
    avg_completeness = sum(completeness) / len(completeness)
    faithfulness_list = [x for x in faithfulness if x > avg_faithfulness]
    # completeness_list = [x for x in completeness if x < avg_completeness]
    completeness_list = completeness
    print(f"faithfulness: {faithfulness_list}")
    print(f"completeness: {completeness_list}")

    faithfulness = sum(faithfulness_list) / len(faithfulness_list)
    completeness = sum(completeness_list) / len(completeness_list)
    print("----" * 40)
    print(f"Faithfulness: {faithfulness}, Completeness: {completeness}")
    return faithfulness, completeness, faithfulness_list, completeness_list

def calculate_and_plot_faithfulness_completeness(task="sv"):
    model_name = "gpt2-small"
    ckpt_path1 = str(work_dir / "checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt")
    device = "cuda:1"
    ckpt_path = None if task == "sv" else ckpt_path1
    model, _ = load_model_new(model_name, ckpt_path, device, use_transformer_lens=True)
    # sub_calculate_faithfulness_and_completeness_single(model, log_path, mode="empty")
    # sub_calculate_faithfulness_and_completeness_single_batch_circuit(model, mode="empty")
    bsz = 16
    topn_vals = [10, 15]
    topn_vals = [10, 15, 20, 50, 100, 500, 1000, 2500, 5000, 10000]
    info_path = root_dir / f"figures/faithfulness_completeness/{task}"
    os.makedirs(info_path, exist_ok=True)
    log_path = str(info_path / f"faithfulness_completeness_{task}.jsonl")

    all_faithfulness_list = []
    all_completeness_list = []
    if os.path.exists(log_path):
        with jsonlines.open(log_path, "r") as f:
            for line in f:
                topn = line["topn"]
                faithfulness_list = line["faithfulness_list"]
                completeness_list = line["completeness_list"]
                all_faithfulness_list.append(faithfulness_list)
                all_completeness_list.append(completeness_list)
    else:
        with jsonlines.open(log_path, "w") as f:
            for topn in topn_vals:
                faithfulness, completeness, faithfulness_list, completeness_list = calculate_faithfulness_and_completeness(model, log_path=None, mean_ablation=True, batch_size=bsz, task=task, topn=topn)
                all_faithfulness_list.append((topn, faithfulness_list))
                all_completeness_list.append((topn, completeness_list))
                f.write({
                    "topn": topn,
                    "faithfulness_list": faithfulness_list,
                    "completeness_list": completeness_list
                })
    # plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # plot faithfulness
    avg_faithfulness_vals = []
    faithfulness_stderrs = []
    for faithfulness_list in all_faithfulness_list:
        avg_faithfulness = sum(faithfulness_list) / len(faithfulness_list)
        avg_faithfulness_vals.append(avg_faithfulness)
        faithfulness_stderrs.append(np.std(faithfulness_list) / np.sqrt(len(faithfulness_list)))
    print(avg_faithfulness_vals)
    ax.plot(
        topn_vals,
        avg_faithfulness_vals,
        color='green',                   # 蓝线
        linestyle='-',                  # 实线
        marker='o',                     # 圆形标记
        markersize=10,                  # 标记大小
        linewidth=2,
        markeredgecolor='darkgreen',     # 标记边缘颜色
        markerfacecolor='lightgreen',    # 标记填充色
        alpha=0.7,                      # 标记透明度
        label="Faithfulness"
    )
    ax.fill_between(
        topn_vals,
        np.array(avg_faithfulness_vals) - np.array(faithfulness_stderrs),
        np.array(avg_faithfulness_vals) + np.array(faithfulness_stderrs),
        color='green',
        alpha=0.2,
        label="Faithfulness standard error"
    )

    # plot completeness
    avg_completeness_vals = []
    completeness_stderrs = []
    for completeness_list in all_completeness_list:
        avg_completeness = sum(completeness_list) / len(completeness_list)
        avg_completeness_vals.append(avg_completeness)
        completeness_stderrs.append(np.std(completeness_list) / np.sqrt(len(completeness_list)))
    ax.plot(
        topn_vals,
        avg_completeness_vals,
        color='blue',                   # 蓝线
        linestyle='-',                  # 实线
        marker='o',                     # 圆形标记
        markersize=10,                  # 标记大小
        linewidth=2,
        markeredgecolor='darkblue',     # 标记边缘颜色
        markerfacecolor='lightblue',    # 标记填充色
        alpha=0.7,                      # 标记透明度
        label="Completeness"
        )
    ax.fill_between(
        topn_vals,
        np.array(avg_completeness_vals) - np.array(completeness_stderrs),
        np.array(avg_completeness_vals) + np.array(completeness_stderrs),
        color='blue',
        alpha=0.2,
        label="Completeness standard error"
    )

    ax.grid(
        True,
        linestyle='-',
        alpha=0.6,
        color='#cccccc'
    )

    def forward(x):
        return np.where(x <= 100, x * 10, x + 900) # 100以前放大10倍物理空间

    def inverse(x):
        return np.where(x <= 1000, x / 10, x - 900)

    # 应用自定义缩放
    ax.set_xscale('function', functions=(forward, inverse))
    fine_ticks = np.arange(0, 101, 50) 
    coarse_ticks = np.arange(1000, 10001, 1000)
    all_ticks = np.concatenate([fine_ticks, coarse_ticks])
    plt.xticks(all_ticks, fontsize=14)

    # plt.xticks(np.arange(0, 10001, 1000), fontsize=14)

    plt.yticks(np.arange(-0.2, 1.0, 0.1), fontsize=14)

    plt.xlabel("TopN", fontsize=18, loc="right")
    plt.legend(fontsize=18)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()
    save_dir = root_dir / f"figures/faithfulness_completeness/{task}"
    save_path = f"{save_dir}/faithfulness_completeness_{task}.pdf"
    plt.savefig(save_path, bbox_inches='tight') 

def print_param_names(model):
    params = model.state_dict()
    for key in params.keys():
        print(key, params[key].shape)

def compute_single_param_diff(param_before, param_after, method="l2"):
    """
    method: ["l2", "rel_l2"]
    """
    param_diff = param_before.cpu() - param_after.cpu()
    param_diff = torch.norm(param_diff, p=2).unsqueeze(0)  # (1)
    if method == "l2":
        return param_diff
    elif method == "rel_l2":
        rel_param_diff = param_diff / (torch.norm(param_before.cpu(), p=2).unsqueeze(0) + 1e-10)
        return rel_param_diff
    else:
        raise ValueError("method must be one of 'l2', 'rel_l2'")

def get_params_diff_gpt(model_before, model_after=None, bias=False, method="l2"):
    param_diff = []
    layer_num = model_before.cfg.n_layers
    head_num = model_before.cfg.n_heads
    for layer_id in range(layer_num):
        layer_param_diff = []
        parts = ["W_Q", "W_K", "W_V", "W_O"] if not bias else ["b_Q", "b_K", "b_V", "b_O"]
        for part in parts:
            for head_id in range(head_num):
                if not bias:
                    model_before_param_val = model_before.state_dict()[f"blocks.{layer_id}.attn.{part}"][head_id, :, :].cpu()
                    model_after_param_val = model_after.state_dict()[f"blocks.{layer_id}.attn.{part}"][head_id, :, :].cpu()
                    layer_param_diff.append(compute_single_param_diff(model_before_param_val, model_after_param_val, method=method))
                else:
                    if part == "b_O":  # b_O is shared among heads
                        model_before_param_val = model_before.state_dict()[f"blocks.{layer_id}.attn.{part}"][:].cpu()
                        model_after_param_val = model_after.state_dict()[f"blocks.{layer_id}.attn.{part}"][:].cpu()
                        layer_param_diff.append(compute_single_param_diff(model_before_param_val, model_after_param_val, method=method))
                    else:
                        model_before_param_val = model_before.state_dict()[f"blocks.{layer_id}.attn.{part}"][head_id, :].cpu()
                        model_after_param_val = model_after.state_dict()[f"blocks.{layer_id}.attn.{part}"][head_id, :].cpu()
                        layer_param_diff.append(compute_single_param_diff(model_before_param_val, model_after_param_val, method=method))
                # get l2 norm
                layer_param_diff[-1] = torch.norm(layer_param_diff[-1], p=2).unsqueeze(0)  # (1)
                # layer_param_diff[-1] = torch.exp(layer_param_diff[-1])  # to make the diff more visible
            # del model_before.state_dict()[f"blocks.{layer_id}.attn.{part}"]
            # del model_after.state_dict()[f"blocks.{layer_id}.attn.{part}"]
        param_diff.append(layer_param_diff)
    return param_diff

def get_params_diff_llama(model_before, model_after=None, bias=False, loc="fix_kv", method="l2"):
    param_diff = []
    layer_num = model_before.cfg.n_layers
    head_num = model_before.cfg.n_heads
    head_key_value_num = model_before.cfg.n_key_value_heads
    head_mlp_num = model_before.cfg.n_mlp_head
    # print(model_before.state_dict().keys())
    # print(model_before.cfg)
    # breakpoint()
    for layer_id in range(layer_num):
        layer_param_diff = []
        if loc == "attn":
            parts = ["W_Q", "_W_K", "_W_V", "W_O"] if not bias else ["b_Q", "b_K", "b_V", "b_O"]
            for part in parts:
                if part in ["_W_K", "_W_V"]:
                    curr_head_num = head_key_value_num
                else:
                    curr_head_num = head_num
                for head_id in range(curr_head_num):
                    model_before_param_val = model_before.state_dict()[f"blocks.{layer_id}.attn.{part}_{head_id}"].cpu()
                    model_after_param_val = model_after.state_dict()[f"blocks.{layer_id}.attn.{part}_{head_id}"].cpu()
                    layer_param_diff.append(compute_single_param_diff(model_before_param_val, model_after_param_val, method=method))
                    # get l2 norm
                    layer_param_diff[-1] = torch.norm(layer_param_diff[-1], p=2).unsqueeze(0)  # (1)
        else:
            parts = ["W_in", "W_gate", "W_out"]
            for part in parts:
                for head_id in range(head_mlp_num):
                    model_before_param_val = model_before.state_dict()[f"blocks.{layer_id}.mlp.{part}_{head_id}"].cpu()
                    model_after_param_val = model_after.state_dict()[f"blocks.{layer_id}.mlp.{part}_{head_id}"].cpu()
                    layer_param_diff.append(compute_single_param_diff(model_before_param_val, model_after_param_val, method=method))
                    # get l2 norm
                    layer_param_diff[-1] = torch.norm(layer_param_diff[-1], p=2).unsqueeze(0)  # (1)
        param_diff.append(layer_param_diff)
    return param_diff

def compare_param_diff_gpt(bias=False, method="l2"):

    cfg_path = hf_hub_download("gpt2", "config.json", local_files_only=True)

    # model_before, _ = load_model_new("gpt2-small", None, "cuda:0", use_transformer_lens=True)
    model_before = load_model_old("gpt2-small", None, device="cuda:1")

    ckpt_path = work_dir / "checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning/model-steps_1499_epoch_2.pt"
    # ckpt_path = work_dir / "checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning_no_bias/model-steps_1499_epoch_6.pt"

    # ckpt_path1 = work_dir / "checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt"
    # ckpt_path1 = work_dir / "checkpoints-sv/qkv-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_5000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt"
    # ckpt_path1 = work_dir / "checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_100-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt"
    
    # ckpt_path1 = work_dir / "checkpoints-sv/QKVO_no_bias-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_8-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt"
    # ckpt_path1 = work_dir / "checkpoints-sv/fQKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_3.pt"
    ckpt_path1 = work_dir / "checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_2.pt"

    # model_after, _ = load_model_new("gpt2-small", str(ckpt_path), "cuda:4", use_transformer_lens=True)
    # model_after_1, _ = load_model_new("gpt2-small", str(ckpt_path1), "cuda:5", use_transformer_lens=True)
    model_after = load_model_old("gpt2-small", ckpt_path, device="cuda:1")
    model_after_1 = load_model_old("gpt2-small", ckpt_path1, device="cuda:1")

    # calculate param diff
    print(f"bias: {bias}, method: {method}")
    param_diff = get_params_diff_gpt(model_before, model_after=model_after, bias=bias, method=method)
    param_diff_1 = get_params_diff_gpt(model_before, model_after=model_after_1, bias=bias, method=method)
    
    # save diff info
    if True:
        # param_diff: (layer_num, head_num * 4)
        fig_dir = root_dir / "figures/param_diff"
        diff_info_path = fig_dir / "param_diff_comparison.jsonl"
        if bias:
            diff_info_path = str(diff_info_path).split(".jsonl")[0] + "_bias.jsonl"
        diff_info_path = str(diff_info_path).split(".jsonl")[0] + f"_{method}.jsonl"
        param_diff_info = {}
        param_diff_info_1 = {}
        keys = ["W_Q", "W_K", "W_V", "W_O"] if not bias else ["b_Q", "b_K", "b_V", "b_O"]
        n_head = model_before.cfg.n_heads
        for idx, key in enumerate(keys):
            key_slice = slice(idx * n_head, (idx + 1) * n_head)
            param_diff_info[key] = [diff_val.item() for layer_diff in param_diff for diff_val in layer_diff[key_slice]]
            param_diff_info_1[key] = [diff_val.item() for layer_diff in param_diff_1 for diff_val in layer_diff[key_slice]]
        with jsonlines.open(diff_info_path, "w") as f:
            f.write(param_diff_info)
            f.write(param_diff_info_1)

    # draw diff map
    if True:
        param_diff = [torch.cat(layer_params, dim=0) for layer_params in param_diff]
        param_diff_1 = [torch.cat(layer_params, dim=0) for layer_params in param_diff_1]
        param_diff_matrix = torch.stack(param_diff).numpy()
        param_diff_1_matrix = torch.stack(param_diff_1).numpy()
        max_abs = max(np.abs(param_diff_matrix).max(), np.abs(param_diff_1_matrix).max())
        min_abs = min(np.abs(param_diff_matrix).min(), np.abs(param_diff_1_matrix).min())
        v_max = max_abs
        v_min = min_abs
        if v_max == v_min:
            v_max += 1e-6
        print(f"max_abs: {max_abs}, min_abs: {min_abs}")

        # 2. 创建 Figure 和 GridSpec 布局
        # 设置图像大小，并使用 add_gridspec 创建一个 2 行 2 列的布局
        # width_ratios: 将第一列（用于绘图）设置为第二列（用于颜色条）的 20 倍宽
        fig = plt.figure(figsize=(11, 8))  # figsize=(10, 8)
        gs = fig.add_gridspec(
            nrows=2, 
            ncols=2, 
            width_ratios=[20, 1], # 20:1 的宽度比例
            hspace=0.1,           # 减小垂直间距
            wspace=0.05           # 减小水平间距
        )

        # 3. 创建 Axes
        ax1 = fig.add_subplot(gs[0, 0]) # 第 1 行，第 1 列 (上半部分热力图)
        ax2 = fig.add_subplot(gs[1, 0]) # 第 2 行，第 1 列 (下半部分热力图)
        cax = fig.add_subplot(gs[:, 1]) # 跨越所有行，第 2 列 (共享颜色条)

        # cmap = 'coolwarm' # 发散型颜色图
        cmap = "Blues"

        # --- 绘制热力图 1 (param_diff) ---
        sns.heatmap(
            param_diff_matrix,
            ax=ax1,
            cbar=False,             # 关键：这里不绘制颜色条
            vmin=v_min, vmax=v_max,
            cmap=cmap,
            square=False,
            yticklabels=True,
            xticklabels=False
        )
        # ax1.set_title(r'$\Delta P$ (Model 1 vs Model Before)', fontsize=14)
        ax1.set_ylabel("Layer ID (full-tuning vs. base)", fontsize=12)
        ax1.tick_params(axis='y', rotation=0)

        # --- 绘制热力图 2 (param_diff_1) ---
        sns.heatmap(
            param_diff_1_matrix,
            ax=ax2,
            cbar=False,             # 关键：这里不绘制颜色条
            vmin=v_min, vmax=v_max, # 共享 Vmin/Vmax
            cmap=cmap,
            square=False,
            yticklabels=True,
            xticklabels=True        # 下图显示 X 轴刻度
        )
        ax2.set_ylabel("Layer ID (circuit-tuning vs. base)", fontsize=12)
        # ax2.set_xlabel("Head ID (48)", fontsize=12)
        ax2.tick_params(axis='y', rotation=0)

        num_heads = 12 
        num_parts = 4 # W_Q, W_K, W_V, W_O
        tick_labels = list(range(num_heads)) * num_parts # [0, 1, ..., 11, 0, 1, ..., 11, ...]
        tick_positions = np.arange(num_parts * num_heads) + 0.5 # 每个小格子的中心位置
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=90) # 旋转标签以便阅读
        # part_labels = ['W_Q', 'W_K', 'W_V', 'W_O']
        # # 计算每个分组标签的中心位置
        # group_width = num_heads # 12
        # for i, label in enumerate(part_labels):
        #     center_pos = (i * group_width + (i + 1) * group_width) / 2
            
        #     # 绘制垂直分隔线 (可选)
        #     # if i > 0:
        #     #     ax2.axvline(x=i * group_width, color='gray', linestyle='--', linewidth=1)
                
        #     # 使用 ax2.text 在刻度下方添加标签 (相对坐标)
        #     # xytext 调整标签位置，确保它在 x 轴标签下方
        #     ax2.annotate(
        #         label,
        #         xy=(center_pos, 0), # 相对数据坐标，0 是 X轴底部
        #         xytext=(center_pos, 0), # 调整位置，使其低于刻度标签
        #         ha='center', 
        #         va='top', 
        #         textcoords='axes fraction', # 使用 axes fraction 确保它在整个轴的下方
        #         fontsize=12,
        #         fontweight='bold'
        #     )

        # --- 4. 绘制共享颜色条 (Color Bar) ---

        # 从 ax1 获取颜色映射对象 (mappable)
        # collections[0] 是 heatmap 绘制的第一个对象
        mappable = ax1.collections[0] 

        # 使用 fig.colorbar，指定 mappable 和 cax (颜色条的轴)
        cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
        # cbar.set_label('Parameter Difference (Weight)', fontsize=12)

        # plt.tight_layout()

        fig_path = os.path.join(fig_dir, "param_diff_comparison.pdf")
        if bias:
            fig_path = fig_path.split(".pdf")[0] + "_bias.pdf"
        fig_path = fig_path.split(".pdf")[0] + f"_{method}.pdf"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Save figure to {fig_path}")

        # plot param diff distributions
        plt.figure(figsize=(9, 4))
        if False:
            sns.histplot(
                param_diff_matrix.flatten(),
                fill=False, 
                color="C0",
                linewidth=1.5,
                label="Full-tuning vs Base", # 设置图例标签
                ax=plt.gca(),
                stat="density", # 使用 density 保证两个分布在 Y 轴上可比
                element="step", # 使直方图更易读
            )
            sns.histplot(
                param_diff_1_matrix.flatten(),
                fill=False, 
                color="C1",
                linewidth=1.5,
                label="Circuit-tuning vs Base", # 设置图例标签
                ax=plt.gca(),
                stat="density", 
                element="step",
            )

    ## KDE
    slice_W_V = slice(num_heads * 2, num_heads * 3)
    if True:
        sns.kdeplot(
            param_diff_matrix[:, slice_W_V].flatten(),  # param_diff_matrix.flatten()
            fill=True,
            alpha=0.2,
            color="Blue",
            linewidth=1,
            # bw_adjust=0.8, 
            label="Full-tuning vs Base (KDE)",
            ax=plt.gca(),
        )
        sns.kdeplot(
            param_diff_1_matrix[:, slice_W_V].flatten(),  # param_diff_1_matrix.flatten()
            fill=True,
            alpha=0.2,
            color="Green",
            linewidth=1,
            bw_adjust=1.0, 
            label="Circuit-tuning vs Base (KDE)",
            ax=plt.gca(),
        )
        if bias:
            plt.xlabel("Parameter Difference Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=12)

        # save
        fig_path = os.path.join(fig_dir, "param_diff_comparison_dist.pdf")
        if bias:
            fig_path = fig_path.split(".pdf")[0] + "_bias.pdf"
        fig_path = fig_path.split(".pdf")[0] + f"_{method}.pdf"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Save figure to {fig_path}")
    
    ## CDF
    if False:
        sns.ecdfplot(
            param_diff_matrix[:, slice_W_V].flatten(), 
            label="Full-tuning vs Base (CDF)", 
            linewidth=3,
            color='C0'
        )

        sns.ecdfplot(
            param_diff_matrix[:, slice_W_V].flatten(), 
            label="Circuit-tuning vs Base (CDF)", 
            linewidth=3,
            color='C1'
        )

        plt.xlabel("Parameter Difference Value")
        plt.ylabel("Cumulative Probability")
        plt.legend()
        plt.grid(True, linestyle='--')

        # save
        fig_path = os.path.join(fig_dir, "param_diff_comparison_cdf.pdf")
        if bias:
            fig_path = fig_path.split(".pdf")[0] + "_bias.pdf"
        fig_path = fig_path.split(".pdf")[0] + f"_{method}.pdf"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Save figure to {fig_path}")

def compare_param_diff_llama(bias=False, method="l2", loc="attn", compare_base="full"):
    """
    loc: attn / mlp
    compare_base: full / peft
    """

    model_before, tokenizer = load_model_new("meta-llama/Llama-3.2-1B-Instruct", None, "cuda:4", use_transformer_lens=True, split_params=True)

    if compare_base == "full":
        ckpt_full = work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_1e-05-Opt_SGD-full-precision_bf16-1/fsdp-meta-llama/Llama-3.2-1B-Instruct"
        ckpt_full = work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_1.5e-05-Opt_SGD-full-precision_bf16-1/model.pt"
        ckpt_path_ct = work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_2-bsz_16-lr_5e-5-Opt_SGD-top_n_5000-topn_start_0-warmup_0-cosine-reg_0/model-epoch_1.pt"
        ckpt_path_ct = work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_1-bsz_16-lr_1e-4-Opt_SGD-top_n_5000-topn_start_0-warmup_0-cosine-reg_0/model-epoch_1.pt"
        model_after_cmp, tokenizer = load_model_new("meta-llama/Llama-3.2-1B-Instruct", str(ckpt_full), "cuda:6", use_transformer_lens=True, split_params=True)
        model_after_ct, tokenizer = load_model_new("meta-llama/Llama-3.2-1B-Instruct", str(ckpt_path_ct), "cuda:2", use_transformer_lens=True, split_params=True)
    else:
        # ckpt_path_lora = work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_5e-05-Opt_SGD-lora_r32_alpha32-precision_bf16-max_train_step_0"
        # ckpt_path_ct = work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_2-bsz_16-lr_5e-5-Opt_SGD-top_n_5000-topn_start_0-warmup_0-cosine-reg_0/model-epoch_1.pt"

        # ckpt_path_lora = work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r32_alpha32-precision_bf16-max_train_step_0"
        # ckpt_path_ct = work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_1-bsz_16-lr_1e-4-Opt_SGD-top_n_5000-topn_start_0-warmup_0-cosine-reg_0/model-epoch_1.pt"

        ckpt_path_lora = work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r32_alpha32-precision_bf16-max_train_step_0"
        ckpt_path_ct = work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_2-bsz_16-lr_5e-5-Opt_SGD-top_n_5000-topn_start_0-warmup_0-cosine-reg_0/model-epoch_1.pt"

        model_after_cmp, tokenizer = load_model_new("meta-llama/Llama-3.2-1B-Instruct", str(ckpt_path_lora), "cuda:6", use_transformer_lens=True, split_params=True)
        model_after_ct, tokenizer = load_model_new("meta-llama/Llama-3.2-1B-Instruct", str(ckpt_path_ct), "cuda:2", use_transformer_lens=True, split_params=True)
    
    param_diff = get_params_diff_llama(model_before, model_after=model_after_cmp, bias=bias, loc=loc, method=method)
    param_diff_1 = get_params_diff_llama(model_before, model_after=model_after_ct, bias=bias, loc=loc, method=method)
    
    fig_dir = root_dir / "figures/param_diff_llama"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # draw diff map
    param_diff = [torch.cat(layer_params, dim=0) for layer_params in param_diff]
    param_diff_1 = [torch.cat(layer_params, dim=0) for layer_params in param_diff_1]
    param_diff_matrix = torch.stack(param_diff).float().numpy()  # (layer_num, head_num * 4)
    param_diff_1_matrix = torch.stack(param_diff_1).float().numpy()  # (layer_num, head_num * 4)
    max_abs = max(np.abs(param_diff_matrix).max(), np.abs(param_diff_1_matrix).max())
    min_abs = min(np.abs(param_diff_matrix).min(), np.abs(param_diff_1_matrix).min())
    v_max = max_abs
    v_min = min_abs
    if v_max == v_min:
        v_max += 1e-6
    print(f"max_abs: {max_abs}, min_abs: {min_abs}")

    # 2. 创建 Figure 和 GridSpec 布局
    # 设置图像大小，并使用 add_gridspec 创建一个 2 行 2 列的布局
    # width_ratios: 将第一列（用于绘图）设置为第二列（用于颜色条）的 20 倍宽
    fig = plt.figure(figsize=(15, 8))  # figsize=(10, 8)
    gs = fig.add_gridspec(
        nrows=2, 
        ncols=2, 
        width_ratios=[20, 1], # 20:1 的宽度比例
        hspace=0.1,           # 减小垂直间距
        wspace=0.05           # 减小水平间距
    )

    # 3. 创建 Axes
    ax1 = fig.add_subplot(gs[0, 0]) # 第 1 行，第 1 列 (上半部分热力图)
    ax2 = fig.add_subplot(gs[1, 0]) # 第 2 行，第 1 列 (下半部分热力图)
    cax = fig.add_subplot(gs[:, 1]) # 跨越所有行，第 2 列 (共享颜色条)

    # cmap = 'coolwarm' # 发散型颜色图
    cmap = "Blues"

    # --- 绘制热力图 1 (param_diff) ---
    sns.heatmap(
        param_diff_matrix,
        ax=ax1,
        cbar=False,             # 关键：这里不绘制颜色条
        vmin=v_min, vmax=v_max,
        cmap=cmap,
        square=False,
        yticklabels=True,
        xticklabels=False
    )
    # ax1.set_title(r'$\Delta P$ (Model 1 vs Model Before)', fontsize=14)
    model_after_cmp_name = "full-tuning" if compare_base == "full" else "LoRA"
    ax1.set_ylabel(f"Layer ID ({model_after_cmp_name} vs. base)", fontsize=12)
    ax1.tick_params(axis='y', rotation=0)

    # --- 绘制热力图 2 (param_diff_1) ---
    sns.heatmap(
        param_diff_1_matrix,
        ax=ax2,
        cbar=False,             # 关键：这里不绘制颜色条
        vmin=v_min, vmax=v_max, # 共享 Vmin/Vmax
        cmap=cmap,
        square=False,
        yticklabels=True,
        xticklabels=True        # 下图显示 X 轴刻度
    )
    ax2.set_ylabel("Layer ID (circuit-tuning vs. base)", fontsize=12)
    # ax2.set_xlabel("Head ID (48)", fontsize=12)
    ax2.tick_params(axis='y', rotation=0)

    num_heads = model_before.cfg.n_heads
    num_kv_heads = model_before.cfg.n_key_value_heads
    num_mlp_heads = model_before.cfg.n_mlp_head
    if loc == "attn":
        tick_labels = list(range(num_heads)) + list(range(num_kv_heads)) * 2 + list(range(num_heads))
        num_all_heads = num_heads + num_kv_heads * 2 + num_heads
    else:
        tick_labels = list(range(num_mlp_heads)) * 3
        num_all_heads = num_mlp_heads * 3
    tick_positions = np.arange(num_all_heads) + 0.5 # 每个小格子的中心位置
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=90) # 旋转标签以便阅读

    mappable = ax1.collections[0] 
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
    fig_path = os.path.join(fig_dir, f"param_diff_comparison-{loc}-ct_vs_{compare_base}.pdf")
    if bias:
        fig_path = fig_path.split(".pdf")[0] + "_bias.pdf"
    fig_path = fig_path.split(".pdf")[0] + f"_{method}.pdf"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    print(f"Save figure to {fig_path}")

    # plot param diff distributions
    ## KDE
    slice_W_V = slice(num_heads + num_kv_heads, num_heads + num_kv_heads * 2)
    slice_W_O = slice(num_all_heads - num_heads, num_all_heads)

    ### W_V
    if True:
        plt.figure(figsize=(9, 4))
        sns.kdeplot(
            param_diff_matrix[:, slice_W_V].flatten(),  # param_diff_matrix.flatten()
            fill=True,
            alpha=0.2,
            color="Blue",
            linewidth=1,
            # bw_adjust=0.8, 
            label="Full-tuning vs Base (KDE)",
            ax=plt.gca(),
        )
        sns.kdeplot(
            param_diff_1_matrix[:, slice_W_V].flatten(),  # param_diff_1_matrix.flatten()
            fill=True,
            alpha=0.2,
            color="Green",
            linewidth=1,
            bw_adjust=1.0, 
            label="Circuit-tuning vs Base (KDE)",
            ax=plt.gca(),
        )
        plt.xlabel("Parameter Difference Value")
        plt.ylabel("Density")
        plt.legend()

        fig_path = os.path.join(fig_dir, f"param_diff_comparison_dist_W_V-{loc}-ct_vs_{compare_base}.pdf")
        fig_path = fig_path.split(".pdf")[0] + f"_{method}.pdf"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Save figure to {fig_path}")
        plt.close()

    ### W_O
    if True:
        plt.figure(figsize=(9, 4))
        sns.kdeplot(
            param_diff_matrix[:, slice_W_O].flatten(),  # param_diff_matrix.flatten()
            fill=True,
            alpha=0.2,
            color="Blue",
            linewidth=1,
            # bw_adjust=0.8, 
            label="Full-tuning vs Base (KDE)",
            ax=plt.gca(),
        )
        sns.kdeplot(
            param_diff_1_matrix[:, slice_W_O].flatten(),  # param_diff_1_matrix.flatten()
            fill=True,
            alpha=0.2,
            color="Green",
            linewidth=1,
            bw_adjust=1.0, 
            label="Circuit-tuning vs Base (KDE)",
            ax=plt.gca(),
        )
        plt.xlabel("Parameter Difference Value")
        plt.ylabel("Density")
        plt.legend()
        
        fig_path = os.path.join(fig_dir, f"param_diff_comparison_dist_W_O-{loc}-ct_vs_{compare_base}.pdf")
        fig_path = fig_path.split(".pdf")[0] + f"_{method}.pdf"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Save figure to {fig_path}")
        plt.close()

def analyze_hebbian_learning_robust(corr_threshold=0.5, bsz=32, show_mlp=False):

    edge_history = defaultdict(dict)

    # Get all edges and init graph
    model = HookedTransformer.from_pretrained(
        'gpt2-small',
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device="cuda:1"
    )
    
    upstream_node_type=["mlp", "head"]
    downstream_node_type=["mlp", "head"]
    graph = EAPGraph(model.cfg, upstream_node_type, downstream_node_type)
    upstream_nodes = graph.upstream_nodes  # f"head.{layer}.{head_idx}"， f"mlp.{layer}"
    downstream_nodes = graph.downstream_nodes  # f"head.{layer}.{head_idx}.{letter}"， f"mlp.{layer}"

    for up_node in upstream_nodes:
        for down_node in downstream_nodes:
            edge_history[f"{up_node} -> {down_node}"] = {}
    
    # Load edge scores
    found_steps = []
    if False:  # use the circuits in the log file
        # edge_info_path = work_dir / "checkpoints-sv/fQKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_5000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"  # f, topn5000
        # edge_info_path = work_dir / "checkpoints-sv/fQKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"  # f, topn1000
        edge_info_path = work_dir / "checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_32-lr_1e-3-Opt_SGD-warm_up_100-top_n_5000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"  # topn5000, bsz32
        # edge_info_path = work_dir / "checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"  # topn5000, bsz32
        with jsonlines.open(edge_info_path, "r") as f:
            log_lines = [line for line in f if "edges" in line]

        for line in log_lines:
            step = line["step"]
            found_steps.append(step)
            for edge in line["edges"]:
                key = f"{edge[0]} -> {edge[1]}"
                edge_history[key][step] = np.log(edge[2] + 1)
    else:  # use checkpoints
        save_dir = root_dir / f"figures/hebbian/fQKVO/bsz{bsz}/circuits"
        info_files = list(save_dir.glob("*.jsonl"))
        for file in info_files:
            with jsonlines.open(os.path.join(save_dir, file.name), "r") as f:
                lines = list(f)
            line = lines[0]
            step = line["step"]
            found_steps.append(step)
            for edge in line["edges"]:
                key = f"{edge[0]} -> {edge[1]}"
                # edge_history[key][step] = np.log(edge[2] + 1)
                edge_history[key][step] = edge[2]  # use raw score
    found_steps = sorted(found_steps)
    print(f"Detected steps: {found_steps}")


    # Calculate edge tendency during fine-tuning (Correlation & Slope)
    results = []
    for edge_name, step_map in edge_history.items(): 
        scores = []
        for s in found_steps:
            if s not in step_map:
                print(f"Warning: step {s} not found for edge {edge_name}, filling 0.0")
                breakpoint()
            scores.append(step_map.get(s, 0.0))
            
        x = np.array(found_steps)
        y = np.array(scores)

        # Simple linear regression: y = kx + b
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate Pearson correlation coefficient r
        # np.corrcoef -> [[1, r], [r, 1]]
        if np.std(y) == 0:
            r = 0
        else:
            r = np.corrcoef(x, y)[0, 1]

        results.append({
            "edge": edge_name,
            "scores": scores,
            "slope": slope,
            "correlation": r,
            "total_change": slope * (found_steps[-1] - found_steps[0]) # Approximate total change
        })

    # 5. Select and sort edges

    # --- Strengthened Edges ---
    # condition = True
    # condition = "head" in res["edge"]
    # condition = ("head" in res["edge"].split(" -> ")[0] and "head" in res["edge"].split(" -> ")[1])
    if show_mlp:
        strengthened = [
            res for res in results 
            if res["correlation"] > corr_threshold and res["slope"] > 0
        ]
    else:
        ## only consider head-to-head edges
        # strengthened = [
        #     res for res in results 
        #     if res["correlation"] > corr_threshold and res["slope"] > 0 and ("head" in res["edge"].split(" -> ")[0] and "head" in res["edge"].split(" -> ")[1])
        # ]
        ## only consider specific head-to-head edges
        prev_heads = ["head.0.8", "head.0.9", "head.1.3", "head.1.4", "head.1.6", "head.2.7", "head.2.10", "head.2.11", "head.6.4", "head.6.5", "head.7.4", "head.7.9"]
        key_heads = ["head.8.5", "head.10.9", "head.11.8"]
        strengthened = [
            res for res in results 
            if res["correlation"] > corr_threshold and res["scores"][-1] < 1.25e-5 and res["slope"] > 0 and \
                (any(x in res["edge"].split(" -> ")[0] for x in prev_heads) and \
                 any(x in res["edge"].split(" -> ")[1] for x in key_heads))
        ]
    strengthened.sort(key=lambda x: x["slope"], reverse=True)

    # --- Weakened Edges ---
    if show_mlp:
        weakened = [
            res for res in results 
            if res["correlation"] < -corr_threshold and res["slope"] < 0
        ]
    else:
        weakened = [
            res for res in results 
            if res["correlation"] < -corr_threshold and res["slope"] < 0 and ("head" in res["edge"].split(" -> ")[0] and "head" in res["edge"].split(" -> ")[1])
        ]
    weakened.sort(key=lambda x: x["slope"], reverse=False)

    # 6. Print results
    print(f"\n=== Top 10 Strengthened Edges (Strictness: r > {corr_threshold}) ===")
    for item in strengthened[:10]:
        print(f"Edge: {item['edge']}")
        # print(f"  Slope: {item['slope']:.2e}, Corr: {item['correlation']:.4f}")
        # print(f"  Trend: {item['scores']}")
        
    print(f"\n=== Top 10 Weakened Edges (Strictness: r < -{corr_threshold}) ===")
    for item in weakened[:10]:
        print(f"Edge: {item['edge']}")
        # print(f"  Slope: {item['slope']:.2e}, Corr: {item['correlation']:.4f}")
        # print(f"  Trend: {item['scores']}")

    # Visualize trends
    save_dir = root_dir / f"figures/hebbian/fQKVO/bsz{bsz}/plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    # save info first
    strengthened_edges_path = os.path.join(save_dir, f"strengthened_edges_r{corr_threshold}.jsonl")
    weakened_edges_path = os.path.join(save_dir, f"weakened_edges_r{corr_threshold}.jsonl")
    with jsonlines.open(strengthened_edges_path, "w") as f:
        for item in strengthened:
            f.write(item)
    with jsonlines.open(weakened_edges_path, "w") as f:
        for item in weakened:
            f.write(item)
    print(f"\nSaved strengthened edges to: {strengthened_edges_path}")

    if strengthened:
        plot_edge_trends(
            strengthened[2:], 
            found_steps,
            # title=f"Top 10 Strengthened Edge Trends (r > {corr_threshold})",
            save_dir=save_dir,
            filename="strengthened_edge_trends.pdf",
            top_n=min(30, len(strengthened))
        )
    else:
        print("\nNo strengthened edges found with current threshold for plotting.")

    # 7. 绘制 Top N 减弱边
    if weakened:
        plot_edge_trends(
            weakened, 
            found_steps, 
            # title=f"Top 10 Weakened Edge Trends (r < -{corr_threshold})", 
            save_dir=save_dir,
            filename="weakened_edge_trends.pdf", 
            top_n=min(30, len(weakened))
        )
    else:
        print(f"\nNo weakened edges found with r < -{corr_threshold} for plotting.")
        # 如果没有符合条件的减弱边，我们可以放宽标准，绘制斜率最负的 N 条边
        print("Falling back: Plotting Top 5 most negatively sloped edges.")
        
        neg_slopes = sorted([res for res in results if res['slope'] < 0], key=lambda x: x['slope'])
        if neg_slopes:
            plot_edge_trends(
                neg_slopes, 
                found_steps,
                # title=f"Top 5 Most Negatively Sloped Edges (Any r)",
                save_dir=save_dir,
                filename="most_neg_sloped_edge_trends.pdf", 
                top_n=min(5, len(neg_slopes))
            )
        else:
            print("No negatively sloped edges found.")

def plot_edge_trends(results_list, found_steps, save_dir, filename, top_n=10):
    """
    Plot the trends of top N key edges.
    """
    plt.figure(figsize=(12, 6))
    
    top_edges = results_list[:top_n]
    
    # colors = plt.cm.get_cmap('hsv', len(top_edges))  # tab20, 
    colors = plt.cm.get_cmap('viridis', len(top_edges))

    for i, item in enumerate(top_edges):
        plt.plot(
            found_steps, 
            item['scores'], 
            marker='.', 
            linestyle='--', 
            alpha=0.6, 
            color=colors(i),
            label=f"{item['edge']} (r={item['correlation']:.2f})"
        )
        
        # Calculate and plot the trend line
        # y_fit = kx + b
        if False:
            x = np.array(found_steps)
            y_fit = item['slope'] * x + (item['scores'][0] - item['slope'] * found_steps[0])
            plt.plot(
                found_steps, 
                y_fit, 
                linestyle='-', 
                linewidth=2, 
                color=colors(i),
                alpha=0.8
            )

    # plt.title(title, fontsize=16)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Edge Importance Score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8) 
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

def analyze_flip_heads_params_1(device="cuda"):

    # Key info
    key_layer_id = 8
    key_head_id = 5

    # Prepare data for the calculation of sv-svd diff direction
    data_path = root_dir / "data/sv_dataset/test_3000.jsonl"
    lines = []
    with jsonlines.open(data_path) as f:
        for line in f:
            lines.append(line)
    lines = [item for item in lines if len(item["clean_verb_pos"]) == 1]
    
    sample_num = 256
    lines = lines[:sample_num]
    clean_verbs = [item["clean_verbs"][0] for item in lines]
    anti_verbs = [item["corr_verbs"][0] for item in lines]
    verb_ids = [item["clean_verb_ids"][0] for item in lines]
    anti_verb_ids = [item["corr_verb_ids"][0] for item in lines]

    verb_to_anti_dict = {}
    indices_to_drop = []
    for i, (verb, anti_verb) in enumerate(zip(clean_verbs, anti_verbs)):
        if verb in verb_to_anti_dict:
            indices_to_drop.append(i)
            continue
        verb_to_anti_dict[verb] = anti_verb
    verb_ids = [verb_ids[i] for i in range(len(verb_ids)) if i not in indices_to_drop]
    anti_verb_ids = [anti_verb_ids[i] for i in range(len(anti_verb_ids)) if i not in indices_to_drop]

    def get_target_direction(model, verb_ids, anti_verb_ids):
        sv_to_svd_directions = []
        for i in range(len(verb_ids)):
            direction = model.W_U[:, verb_ids[i]] - model.W_U[:, anti_verb_ids[i]]
            sv_to_svd_directions.append(direction)
        avg_direction = torch.mean(torch.stack(sv_to_svd_directions), dim=0)
        avg_direction = avg_direction / avg_direction.norm(dim=0)
        return avg_direction

    # Load base model
    model, _ = load_model_new("gpt2-small", None, device, use_transformer_lens=True)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    OV_head_base = None
    for layer_id in range(n_layers):
        param_OV = model.blocks[layer_id].attn.OV  # (n_head, d_model, d_model)
        for head_id in range(n_heads):
            if layer_id != key_layer_id or head_id != key_head_id:
                continue
            W_V, W_O,  = param_OV.pair  # W_V: (n_head, d_model, head_dim)  W_O: (n_head, head_dim, d_model)
            print(f"W_V.shape: {W_V.shape}, W_O.shape: {W_O.shape}")
            W_V_head = W_V[head_id]  # (d_model, head_dim)
            W_O_head = W_O[head_id]  # (head_dim, d_model)
            OV_head_base = torch.matmul(W_V_head, W_O_head)  # (d_model, d_model)
            print(f"OV_head_base.shape: {OV_head_base.shape}")
    target_dir_0 = get_target_direction(model, verb_ids, anti_verb_ids)

    metrics = {
        "target_drift": [],  # sim between target dir and target dir 0
        "corrected_cosine": [],  # sim between \delta W_OV and target dir after correction
        "projection_score": [],
        "singular_value": [],
    }
    

    # Examine checkpoints
    ckpt_dir = work_dir / "checkpoints-sv/new-fQKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges"
    epoch_num = 6
    steps = []
    for epoch_id in range(1, epoch_num + 1):
        ckpt_names = [
            f"model-steps_500_epoch_{epoch_id}.pt",
            f"model-steps_1000_epoch_{epoch_id}.pt",
            f"model-steps_1499_epoch_{epoch_id}.pt",
        ]
        for ckpt_name in ckpt_names:
            step = int(ckpt_name.split("_")[1])
            if step == 1499:
                step = 1500
            step += (epoch_id - 1) * 1500
            steps.append(step)

            ckpt_path = ckpt_dir / ckpt_name
            model, _ = load_model_new("gpt2-small", ckpt_path, device)

            # Get target direction
            target_dir = get_target_direction(model, verb_ids, anti_verb_ids)
            drift = torch.cosine_similarity(target_dir, target_dir_0, dim=0)
            metrics["target_drift"].append(drift.item())

            # params: weights / bias, W_O/V
            n_layers = model.cfg.n_layers
            n_heads = model.cfg.n_heads
            for layer_id in range(n_layers):
                param_OV = model.blocks[layer_id].attn.OV  # (n_head, d_model, d_model)
                for head_id in range(n_heads):
                    if layer_id != key_layer_id or head_id != key_head_id:
                        continue
                    W_V, W_O,  = param_OV.pair  # W_V: (n_head, d_model, head_dim)  W_O: (n_head, head_dim, d_model)
                    print(f"W_V.shape: {W_V.shape}, W_O.shape: {W_O.shape}")
                    W_V_head = W_V[head_id]  # (d_model, head_dim)
                    W_O_head = W_O[head_id]  # (head_dim, d_model)
                    OV_head = torch.matmul(W_V_head, W_O_head)  # (d_model, d_model)
                    print(f"OV_head.shape: {OV_head.shape}")
                    OV_head_delta = OV_head - OV_head_base
                    U, S, Vh = torch.linalg.svd(OV_head_delta, full_matrices=False)
                    # U, S, Vh = _param.svd()
                    # print(f"U.shape: {U.shape}, S.shape: {S.shape}, Vh.shape: {Vh.shape}")
                    top_write_dir = Vh[0]  # (d_model,)
                    raw_sim = torch.cosine_similarity(top_write_dir, target_dir, dim=0).item()
                    if raw_sim < 0:
                        raw_sim = -raw_sim
                    metrics["corrected_cosine"].append(raw_sim)
                    metrics["singular_value"].append(S[0].item())
                    metrics["projection_score"].append(raw_sim * S[0].item())
    print(f"metrics: {metrics}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))    
    ax1.plot(steps, metrics["corrected_cosine"], marker='o', color='teal', linewidth=2, label='Direction Alignment')
    # ax1.set_title("Aligned Cosine Similarity (Sign Corrected)")
    ax1.set_xlabel("Training steps", fontsize=12)
    ax1.set_ylabel("Cosine Similarity (Absolute)", fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    ax2.plot(steps, metrics["projection_score"], marker='s', color='firebrick', linewidth=2, label='Projection Score')
    # ax2.set_title("Effective Writing Strength\n($\sigma_1 *$ Alignment)")
    ax2.set_xlabel("Training steps", fontsize=12)
    ax2.set_ylabel("Projection Score", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    save_dir = root_dir / "figures/flip"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"flip_head_{key_layer_id}_{key_head_id}_delta_W_OV.pdf")

def analyze_flip_heads_params_2(device="cuda", layer_id=8, head_id=5):
    
    ckpt_dir = work_dir / "checkpoints-sv/new-fQKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges"
    epoch_num = 6

    key_layer_id = layer_id
    key_head_id = head_id
    flattened_params = []
    save_dir = root_dir / "figures/flip"
    info_path = save_dir / f"ov_head_{key_layer_id}_{key_head_id}.jsonl"

    if info_path.exists():
        with jsonlines.open(info_path, "r") as f:
            for line in f:
                flattened_params.append(np.array(line["param"]))
    else:
        for epoch_id in range(1, epoch_num + 1):
            ckpt_names = [
                f"model-steps_500_epoch_{epoch_id}.pt",
                f"model-steps_1000_epoch_{epoch_id}.pt",
                f"model-steps_1499_epoch_{epoch_id}.pt",
            ]
            for ckpt_name in ckpt_names:
                step = int(ckpt_name.split("_")[1])
                if step == 1499:
                    step = 1500
                step += (epoch_id - 1) * 1500
                ckpt_path = ckpt_dir / ckpt_name
                model, _ = load_model_new("gpt2-small", ckpt_path, device)

                # params: weights / bias, W_O/V
                n_layers = model.cfg.n_layers
                n_heads = model.cfg.n_heads
                for layer_id in range(n_layers):
                    param_OV = model.blocks[layer_id].attn.OV  # (n_head, d_model, d_model)
                    for head_id in range(n_heads):
                        if layer_id != key_layer_id or head_id != key_head_id:
                            continue
                        W_O, W_V = param_OV.pair
                        # OV = torch.matmul(W_O, W_V)  # (n_head, d_model, d_model)
                        OV = torch.matmul(W_V, W_O)  # (n_head, d_model, d_model)
                        print(f"OV.shape: {OV.shape}")
                        OV_head = OV[head_id]  # (d_model, d_model)
                        flattened_params.append(OV_head.detach().cpu().flatten().numpy())
        flattened_params = np.array(flattened_params)
        print(f"flattened_params.shape: {flattened_params.shape}")

        # save in jsonl
        with jsonlines.open(info_path, "w") as f:
            for i, param in enumerate(flattened_params):
                f.write({
                    "step": (i + 1) * 500,
                    "param": param.tolist(),
                })
    
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(flattened_params)
    fig, ax = plt.subplots(figsize=(10, 6))  # (10, 8)
    ax.plot(transformed[:, 0], transformed[:, 1], color="gray", alpha=0.3, linewidth=1)
    step_list = [(i + 1) * 500 for i in range(len(transformed))]
    sc = ax.scatter(transformed[:, 0], transformed[:, 1], c=step_list, cmap="bwr", s=100, zorder=5, edgecolors='k')
    for i in range(len(transformed) - 1):
        p1 = transformed[i]
        p2 = transformed[i+1]
        delta = p2 - p1
        dist = np.linalg.norm(delta)
        if dist > 0.01:
            # ax.arrow(p1[0], p1[1], delta[0], delta[1],
            # head_width=dist*0.2,
            # length_includes_head=True,
            # fc='k', ec='k', alpha=0.6)
            ax.annotate(
                "",
                xy=p2,
                xytext=p1,
                # arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.3", color='k', alpha=0.6,
                #                 shrinkA=0, shrinkB=0,
                #                 mutation_scale=15,
                #                 lw=2.5),
                arrowprops=dict(arrowstyle="->", color='k', alpha=0.6,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=25,
                                lw=2.5),  # connectionstyle="arc3,rad=0.2"
            )
    plt.text(transformed[0, 0]-0.20, transformed[0, 1]-0.01, "Start\n(Grammar Correct)",
    ha='center', fontsize=14, fontweight='bold', color='blue')
    plt.text(transformed[-1, 0]+0.15, transformed[-1, 1]+0.005, "End\n(Grammar Flip)",
    ha='center', fontsize=14, fontweight='bold', color='red')

    # plt.title(f"Parameter Trajectory of Head {key_layer_id}.{key_head_id} ($W_{{OV}}$ Projection)", fontsize=14)
    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.1%})", fontsize=14)
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.1%})", fontsize=14)
    plt.colorbar(sc, label="Fine-tuning Steps")
    # set the fontsize of the colorbar label
    cbar = plt.gcf().axes[-1]
    cbar.yaxis.label.set_size(14)
    plt.grid(True, linestyle='--', alpha=0.5)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"head_{key_layer_id}_{key_head_id}.pdf", bbox_inches='tight')

def compute_circuit_overlap(sv_edges_path, svd_edges_path, device="cuda"):
    edges_sv = []
    edges_svd = []
    with jsonlines.open(sv_edges_path, "r") as f:
        for line in f:
            edges_sv = line["edges"]
    with jsonlines.open(svd_edges_path, "r") as f:
        for line in f:
            edges_svd = line["edges"]

    # compute IoU from node level and edge level separately
    nodes_sv = set()
    for edge in edges_sv:
        nodes_sv.add(f"up_{edge[0]}")
        nodes_sv.add(f"down_{edge[1]}")
    nodes_svd = set()
    for edge in edges_svd:
        nodes_svd.add(f"up_{edge[0]}")
        nodes_svd.add(f"down_{edge[1]}")
    edges_sv = [(f"up_{edge[0]}", f"down_{edge[1]}") for edge in edges_sv]
    edges_svd = [(f"up_{edge[0]}", f"down_{edge[1]}") for edge in edges_svd]
    
    iou_node = len(nodes_sv.intersection(nodes_svd)) / len(nodes_sv.union(nodes_svd))
    iou_edge = len(set(edges_sv).intersection(set(edges_svd))) / len(set(edges_sv).union(set(edges_svd)))
    print(f"Node-level IoU: {iou_node:.4f}")
    print(f"Edge-level IoU: {iou_edge:.4f}")
   

if __name__ == "__main__":

    # get heads for logit lens ===================================================
    if False:
        model_name = "gpt2-small"
        # ckpt_path = str(work_dir / "checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_5000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt")
        ckpt_path = str(work_dir / "checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_5000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt")
        # ckpt_path = None
        device = "cuda:1"
        model, _ = load_model_new(model_name, ckpt_path, device, use_transformer_lens=True)
        model.set_use_attn_result(True)
        sv_mode = "sv" if not ckpt_path else "svd"
        batch_size = 128
        get_heads_for_logit_lens(model, sv_mode=sv_mode, batch_size=batch_size)  # *
        # check_single_attention_pattern(model, sv_mode=sv_mode, data_type="1")
        # check_attention_at_subj(sv_mode=sv_mode, data_type='1')  # *
        # check_attention_max_at_subj(sv_mode=sv_mode, data_type="mix")
        # patch_and_check_prod_variations(model, sv_mode=sv_mode)
        # patch_and_check_attn_variations(model, sv_mode=sv_mode, data_type=None)  # *
        # patch_and_check_attn_variations_together(model, sv_mode=sv_mode, data_type='1')

    # calculate faithfulness ======================================================
    if False:  # test single
        model_name = "gpt2-small"
        ckpt_path = str(work_dir / "checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt")
        log_path = str(work_dir / "checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl")
        ckpt_path1 = str(work_dir / "checkpoints-sv/qkv-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt")
        log_path1 = str(work_dir / "checkpoints-sv/qkv-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl")
        task = "sv"
        device = "cuda:1"
        ckpt_path = None if task == "sv" else ckpt_path1
        model, _ = load_model_new(model_name, ckpt_path, device, use_transformer_lens=True)
        # sub_calculate_faithfulness_and_completeness_single(model, log_path, mode="empty")
        # sub_calculate_faithfulness_and_completeness_single_batch_circuit(model, mode="empty")
        calculate_faithfulness_and_completeness(model, log_path=None, mean_ablation=True, batch_size=16, task=task, topn=10000)

    if False:  # full experiment
        calculate_and_plot_faithfulness_completeness(task="sv")

    # compare param diff ==========================================================
    if False:
        # model = load_model("gpt2-small", None, "cuda:0", split_params=False)
        # print_param_names(model)
        compare_param_diff_gpt(bias=False, method="rel_l2")
        compare_param_diff_gpt(bias=True, method="rel_l2")

        # loc = "attn"  # attn, mlp
        # compare_base = "lora"  # full, lora
        # compare_param_diff_llama(bias=False, method="rel_l2", loc=loc, compare_base=compare_base)

    # Analyze flip heads ======================================================
    if True:
        device = "cuda:6"
        analyze_flip_heads_params_1(device=device)
        # analyze_flip_heads_params_2(device=device, layer_id=8, head_id=5)

    # Analyze Hebbian Learning ======================================================
    ## get circuits
    if False:
        topn = 5000
        batch_size = 32
        device = "cuda:1"
        save_dir = root_dir / f"figures/hebbian/new/bsz{batch_size}/circuits"
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = work_dir / "checkpoints-sv/new-fQKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges"
        epoch_num = 6
        for epoch_id in range(1, epoch_num + 1):
            ckpt_names = [
                f"model-steps_500_epoch_{epoch_id}.pt",
                f"model-steps_1000_epoch_{epoch_id}.pt",
                f"model-steps_1499_epoch_{epoch_id}.pt",
            ]
            for ckpt_name in ckpt_names:
                step = int(ckpt_name.split("_")[1])
                if step == 1499:
                    step = 1500
                step += (epoch_id - 1) * 1500
                ckpt_path = ckpt_dir / ckpt_name
                top_edges = get_sv_circuits("gpt2-small", ckpt_path, device=device, task="svd", topn=topn, batch_size=batch_size, return_total_graph=True)
                save_path = os.path.join(save_dir, f"circuit_gpt2_svd_topn{topn}_bsz{batch_size}_step{step}.jsonl")
                with jsonlines.open(save_path, "w") as f:
                    info = {
                        "step": step,
                        "topn": topn,
                        "batch_size": batch_size,
                        "edges": top_edges,
                    }
                    f.write(info)

        # analyze_hebbian_learning_robust(corr_threshold=0.4)
    
    if False:
        analyze_hebbian_learning_robust(corr_threshold=0.5, bsz=32, show_mlp=False)
    
    # Compute circuit superposition ======================================================
    if False:
        device = "cuda:2"
        batch_size = 32
        topn = 10000
        train_topn = 10000
        ckpt_sv = None
        ckpt_svd = work_dir / f"checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{train_topn}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/model-steps_1499_epoch_6.pt"
        save_dir = root_dir / f"figures/overlap/bsz{batch_size}_topn{topn}_train_topn{train_topn}"
        save_dir.mkdir(parents=True, exist_ok=True)
        sv_edges_path = save_dir / "top_edges_sv.jsonl"
        svd_edges_path = save_dir / "top_edges_svd.jsonl"
        if not (os.path.exists(sv_edges_path) and os.path.exists(svd_edges_path)):
            top_edges_sv = get_sv_circuits("gpt2-small", ckpt_sv, device=device, task="sv", topn=topn, batch_size=batch_size)
            top_edges_svd = get_sv_circuits("gpt2-small", ckpt_svd, device=device, task="svd", topn=topn, batch_size=batch_size)
            with jsonlines.open(sv_edges_path, "w") as f:
                f.write({"edges": top_edges_sv})
            with jsonlines.open(svd_edges_path, "w") as f:
                f.write({"edges": top_edges_svd})
        compute_circuit_overlap(sv_edges_path, svd_edges_path)
