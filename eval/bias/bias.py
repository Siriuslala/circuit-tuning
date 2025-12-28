"""
Evaluate bias in LLMs.
"""

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import jsonlines
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaConfig
from llama_recipes.model_checkpointing.checkpoint_handler import load_sharded_model_single_gpu
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dist_cp

from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
from peft import get_peft_model, PeftModel

from huggingface_hub import hf_hub_download

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(str(root_dir))

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_model, load_model_from_ckpt, load_model_gpt, load_model_llama, hookdedTF_to_TF
from circuit_data import BiasDataset, BiasCollateFn

MALE_ATTRIBUTES = ["abbot", "actor", "uncle", "baron", "groom", "canary", "son", "emperor", "male", "boy", "boyfriend", "grandson", "heir", "him", "hero", "his", "himself", "host", "gentlemen", "lord", "sir", "manservant""mister", "master", "father", "manny", "nephew", "monk", "priest", "prince", "king", "he", "brother", "tenor", "stepfather", "waiter", "widower", "husband", "man", "men"
]

FEMALE_ATTRIBUTES = ["abbess", "actress", "aunt", "baroness", "bride", "canary", "daughter", "empress", "female", "girl", "girlfriend", "granddaughter", "heiress", "her", "heroine", "hers", "herself", "hostess", "ladies", "lady", "madam", "maid", "miss", "mistress", "mother", "nanny", "niece", "nun", "priestess", "princess", "queen", "she", "sister", "soprano", "stepmother", "waitress", "widow", "wife", "woman", "women"
]

def gen_prompts(model_name):
    if "gpt" in model_name.lower():
        data_path = root_dir / "data/bias/gender_bias/gender_bias_test.jsonl"
    else:
        data_path = root_dir / "data/bias/gender_bias/gender_bias_test_llama.jsonl"
    data = []
    with jsonlines.open(data_path) as f:
        data = [line for line in f]
    return data
    
def forward_pass(model, 
                tokenizer,
                prompts,
                device=None, 
            ):
    
    dataloader = DataLoader(prompts, batch_size=8, shuffle=False)
    
    ppls = []
    losses = []
    losses_reg = []
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on test set", colour="green"):
            pronoun_positions = batch["pronoun_pos"]
            pronoun_ids = batch["pronoun_id"]
            pronoun_anti_ids = batch["pronoun_anti_id"]
            
            inputs = tokenizer(batch["sentence"], 
                                padding=True,
                                padding_side="right",
                                return_tensors='pt',
                                add_special_tokens=False)  
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            if isinstance(model, HookedTransformer):
                logits, loss = model(input=input_ids, attention_mask=attention_mask, return_type="both")
                loss = loss.item()
            else:
                output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = output.logits
                loss = output.loss.cpu() if output.loss is not None else 0
            
            losses.append(loss)       
            ppl = np.exp(loss)
            ppls.append(ppl)
            
            for i in range(len(batch["sentence"])):
                # 是考察 pronoun 的 logit，还是所有 attribute 的 logit？
                target_logits = logits[i, pronoun_positions[i] - 1]
                
                logit_pronoun = target_logits[pronoun_ids[i]].item()
                logit_pronoun_anti = target_logits[pronoun_anti_ids[i]].item()
                losses_reg.append(abs(logit_pronoun - logit_pronoun_anti))
                
                probs = torch.nn.functional.softmax(target_logits, dim=-1)
                prob_pronoun = probs[pronoun_ids[i]].item()
                prob_pronoun_anti = probs[pronoun_anti_ids[i]].item()
                total = prob_pronoun + prob_pronoun_anti
                
                prob_pronoun = prob_pronoun / total
                prob_pronoun_anti = prob_pronoun_anti / total                
                
                stereotype_pronoun = max(prob_pronoun / 0.5 - 1, 0)
                stereotype_pronoun_anti = max(prob_pronoun_anti / 0.5 - 1, 0)
                risk = max(stereotype_pronoun, stereotype_pronoun_anti)
                
                gender = "male" if batch["pronoun"][i] in ["he", "him", "his"] else "female"
                if gender == "male":
                    result = (prob_pronoun, prob_pronoun_anti, risk)
                else:
                    result = (prob_pronoun_anti, prob_pronoun, risk)
                results.append(result)
                
    loss_final = sum(losses) / len(losses)
    ppl_final = np.exp(sum([np.log(ppl) for ppl in ppls]) / len(ppls))
    reg_final = sum(losses_reg) / len(losses_reg)
    
    return results, loss_final, ppl_final, reg_final

def check_data():
    data_path = root_dir / "data/bias/winobias/processed/pro_stereotyped_type2_test.jsonl"
    occupation_set = set()
    occupation_sample_num = defaultdict(int)
    with jsonlines.open(data_path) as f:
        data = [line for line in f]
    for line in data:
        occupation = line["occupation"].replace("the ", "")
        occupation_set.add(occupation)
        occupation_sample_num[occupation] += 1
    print(occupation_set, len(occupation_set))
    print(occupation_sample_num)

def plot_gender_bias_distrubution_old(occupation_risk_dict):
    
    # 对偏见程度进行排序
    sorted_risks = sorted(occupation_risk_dict.values())

    # 计算每个偏见程度下的职业数目
    risks, occupation_num = np.unique(sorted_risks, return_counts=True)

    # 为了插值，我们需要创建一个更细的偏见程度网格
    grid = np.linspace(risks.min(), risks.max(), num=100)

    # 使用线性插值进行平滑处理
    f = interp1d(risks, occupation_num, kind='linear', fill_value="extrapolate")
    interpolated_occupation_num = f(grid)

    # 绘制分布图
    plt.figure(figsize=(10, 6))
    # plt.bar(risks, occupation_num, width=0.05, label='original data')
    plt.plot(grid, interpolated_occupation_num, label='interpolated data', color='green')
    plt.xlabel('Prejudice Risk')
    plt.ylabel('Number of Occupations')
    plt.legend()
    plt.savefig(root_dir / "eval/bias/gender_bias_prejudice_risk_dist.pdf")

def plot_gender_bias_distrubution(occupation_risk_dict, model_name, save_dir):
    
    bias_levels = np.array(list(occupation_risk_dict.values()))

    kde = gaussian_kde(bias_levels)
    bias_grid = np.linspace(bias_levels.min(), bias_levels.max(), 1000)
    density = kde(bias_grid)

    plt.figure(figsize=(10, 6))
    plt.plot(bias_grid, density, label=f'{model_name}', color='green')
    plt.xlabel('Prejudice Risk')
    plt.ylabel('Number of Occupations')
    plt.legend()
    save_path = os.path.join(save_dir, "gender_bias_prejudice_risk_dist.pdf")
    plt.savefig(save_path)

def draw_single_model(model_name="gpt2-small"):
    gpt2_path = root_dir / "eval/results/bias/gpt2-small-debias/result.jsonl"
    gpt2_full_path = root_dir / "eval/results/bias/gpt2-small-debias-lr1e-3-full/result.jsonl"
    gpt2_ct_path = root_dir / "eval/results/bias/gpt2-small-debias-lr1e-3-topn2500/result.jsonl"
    gpt2_ct_reg_path = root_dir / "eval/results/bias/gpt2-small-debias-lr1e-3-topn2500-reg0.5/result.jsonl"
    
    llama_path = root_dir / "eval/results/bias/llama-3.2-1b-it-debias/result.jsonl"
    llama_full_path = root_dir / "eval/results/bias/llama-3.2-1b-it-debias-lr1e-5-full/result.jsonl"
    llama_peft_ct_path = root_dir / "eval/results/bias/llama-3.2-1b-it-debias-peft-lr5e-5-r16-alpha16/result.jsonl"
    llama_ct_path = root_dir / "eval/results/bias/fix_gate-llama-3.2-1b-it-debias-ct-epoch1-lr3e-4-bsz8_accu2-top5000-reg0/result.jsonl"
    llama_ct_reg_path = root_dir / "eval/results/bias/fix_gate-llama-3.2-1b-it-debias-ct-epoch1-lr5e-5-top5000-reg0.4/result.jsonl"
    if "gpt" in model_name:
        ckpt_paths = [
            gpt2_path,
            gpt2_full_path,
            gpt2_ct_path,
            gpt2_ct_reg_path,
        ]
        labels = [
            "GPT2",
            "GPT2-Full-tuning",
            "GPT2-Circuit-Tuning",
            "GPT2-Circuit-Tuning-Regularized",
        ]
    else:
        ckpt_paths = [
            llama_path,
            llama_full_path,
            llama_peft_ct_path,
            llama_ct_path,
            llama_ct_reg_path,
        ]
        labels = [
            "Llama-3.2-1B-it",
            "Llama-3.2-1B-it-Full-tuning",
            "Llama-3.2-1B-it-LoRA",
            "Llama-3.2-1B-it-Circuit-Tuning",
            "Llama-3.2-1B-it-Circuit-Tuning-Regularized",
        ]
    

    bias_grid_list = []
    density_list = []
    for ckpt_path in ckpt_paths:
        prejudice_risk = {}
        with jsonlines.open(ckpt_path, 'r') as f:
            for line in f:
                prejudice_risk = line["occupation_prejudice_risk"]
                break
        
        bias_levels = np.array(list(prejudice_risk.values()))
        bias_grid = np.linspace(bias_levels.min(), bias_levels.max(), 1000)
        density = gaussian_kde(bias_levels)(bias_grid)
        bias_grid_list.append(bias_grid)
        density_list.append(density)
        
    plt.figure(figsize=(10, 6))
    
    # color = ["#35B597", "#005BAC"]
    # color = "#35B597"
    # linestyles = [":", "-.", "--", "-", ""]
    # for i in range(len(ckpt_paths)):
    #     plt.plot(bias_grid_list[i], density_list[i], label=labels[i], color=color, linestyle=linestyles[i])
    
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    N = len(ckpt_paths)
    colors = [cmap(i) for i in np.linspace(0, 1, N)]
    for i in range(N):
        plt.plot(
            bias_grid_list[i], 
            density_list[i], 
            label=labels[i], 
            color=colors[i],
            linestyle='-' 
        )

    plt.xlabel('Discrimination Risk')
    plt.ylabel('Number of Occupations')
    plt.legend()
    
    save_dir = root_dir / "figures/gender_debiasing"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"gender_bias_prejudice_risk_dist_compare_{model_name}.pdf"
    plt.savefig(save_path)
 
def evaluate_bias(model, tokenizer, device, config):
    samples = gen_prompts(config.model)
    prob_results, loss_final, ppl_final, reg_final = forward_pass(model, tokenizer, samples, device)
    
    occupation_bias_dict = defaultdict(list)
    occupation_prejudice_risk_dict = defaultdict(list)
    for sample, prob_result in zip(samples, prob_results):
        occupation = sample["occupation"] 
        bias = prob_result[0] - prob_result[1]  # male - female
        occupation_bias_dict[occupation].append(bias)
        occupation_prejudice_risk_dict[occupation].append(prob_result[2])
        
    occupation_bias_dict = {occupation: np.mean(bias) for occupation, bias in occupation_bias_dict.items()}
    occupation_bias = np.mean(list(occupation_bias_dict.values()))
    occupation_prejudice_risk_dict = {occupation: np.mean(risk) for occupation, risk in occupation_prejudice_risk_dict.items()}
    occupation_prejudice_risk = np.mean(list(occupation_prejudice_risk_dict.values()))
    result = {
        "loss_final": loss_final, 
        "ppl": ppl_final,
        "reg_final": reg_final,
        "occupation_bias": occupation_bias,
        "occupation_prejudice_risk": occupation_prejudice_risk,
    }
    return result

def calculate_prejudice_risk(result_path):
    risk = []
    with jsonlines.open(result_path, 'r') as f:
        for line in f:
            risk = line["occupation_prejudice_risk"]
            break
    avg_risk = np.mean(list(risk.values()))
    print("Average Prejudice Risk:", avg_risk)
    
def main(args):
    
    if "gpt" in args.model_name:
        model, tokenizer = load_model_gpt(args.model_name, args.ckpt_path, args.device)
    else:
        if "Circuit" in args.ckpt_path:
            split_params = True
        else:
            split_params = False
        model, tokenizer = load_model_llama(args.model_name, args.ckpt_path, args.device, split_params=split_params)

    samples = gen_prompts(args.model_name)
    prob_results, loss_final, ppl_final, reg_final = forward_pass(model, tokenizer, samples, args.device)
    
    occupation_bias_dict = defaultdict(list)
    occupation_prejudice_risk_dict = defaultdict(list)
    for sample, prob_result in zip(samples, prob_results):
        occupation = sample["occupation"] 
        bias = prob_result[0] - prob_result[1]  # male - female
        occupation_bias_dict[occupation].append(bias)
        occupation_prejudice_risk_dict[occupation].append(prob_result[2])
        print("male:", prob_result[0], "female:", prob_result[1], "bias:", bias, "risk:", prob_result[2])
        
    occupation_bias_dict = {occupation: np.mean(bias) for occupation, bias in occupation_bias_dict.items()}
    occupation_prejudice_risk_dict = {occupation: np.mean(risk) for occupation, risk in occupation_prejudice_risk_dict.items()}
    # print(occupation_bias_dict)
    # print(occupation_prejudice_risk_dict)
    
    # plot (bias, occupation_num)
    # 是否要根据职业人数进行加权？
    save_dir = os.path.join(args.save_dir, args.output_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plot_gender_bias_distrubution(occupation_prejudice_risk_dict, args.model_name, save_dir)
    result = {
        "occupation_bias": occupation_bias_dict,
        "occupation_prejudice_risk": occupation_prejudice_risk_dict,
    }
    save_path = os.path.join(save_dir, "result.jsonl")
    with jsonlines.open(save_path, "w") as f:
        f.write(result)
        
    calculate_prejudice_risk(save_path)


if __name__ == "__main__":
    
    if False:
        parser = argparse.ArgumentParser()
        if False:
            parser.add_argument("--save_dir", type=str, default=str(root_dir / "eval/results/bias"))
            parser.add_argument("--model_name", type=str, default="gpt2-small")  # "gpt2-small"
            parser.add_argument("--ckpt_path", type=str, default=str(work_dir / "checkpoints-bias/gpt2-small-bias-epochs_3-bsz_16-lr_5e-4-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning/model-steps_964_epoch_2.pt"))
            parser.add_argument("--device", type=str, default="cuda:7")
            parser.add_argument("--output_dir", type=str, default="gpt2-small-debias-lr5e-4-full-epoch_2") # gpt2-small-debias-lr5e-4-regularize0.5-topn5000_warmup500_cosine

            # gpt2-small-bias-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning/model-steps_964_epoch_3.pt
            # fQKVO-gpt2-small-bias-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_2500-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges-reg_beta_0.0/model-steps_964_epoch_3.pt
            # fQKVO-gpt2-small-bias-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_2500-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges-reg_beta_0.5/model-steps_964_epoch_3.pt
                
        # parser.add_argument("--save_dir", type=str, default=str(root_dir / "eval/results/bias"))
        # parser.add_argument("--model_name", type=str, default="llama-3.1-8b-it")
        # parser.add_argument("--ckpt_path", type=str, default=str(work_dir / "checkpoints-bias/llama3.1-8b-it-bias-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r32_alpha64-precision_bf16-max_train_step_500/peft_1"))
        # parser.add_argument("--device", type=str, default="cuda:1")
        # parser.add_argument("--output_dir", type=str, default=str(root_dir / "eval/bias/results/llama-3.1-8b-it-peft")) # 

        if True:
            ckpt = ""
            ckpt_full = str(work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_1.5e-05-Opt_SGD-full-precision_bf16-1/model.pt")
            ckpt_peft = str(work_dir / "checkpoints-bias/llama-3.2-1B-it-bias-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r32_alpha32-precision_bf16-max_train_step_0")
            ckpt_ct = str(work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_1-bsz_16-ori_bsz_8-lr_3e-4-Opt_SGD-top_n_10000-topn_start_0-warmup_0-cosine-reg_0/model-epoch_1.pt")
            ckpt_ct_reg = str(work_dir / "checkpoints-bias/fix_gate-Circuit-Llama-3.2-1B-Instruct-bias-epochs_1-bsz_16-ori_bsz_8-lr_5e-5-Opt_SGD-top_n_5000-topn_start_0-warmup_0-cosine-reg_0.35/model-epoch_1.pt")
            parser.add_argument("--save_dir", type=str, default=str(root_dir) + "/eval/results/bias")
            parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")  # 
            parser.add_argument("--ckpt_path", type=str, default=ckpt_full)
            parser.add_argument("--device", type=str, default="cuda:6")
            parser.add_argument("--output_dir", type=str, default="llama-3.2-1b-it-debias-lr1.5e-5-full")
            # fix_gate-llama-3.2-1b-it-debias-ct-epoch1-lr1e-5-top5000-reg0.5
            
        args = parser.parse_args()
        main(args)
    
    # draw_single_model("gpt2-small")
    draw_single_model("llama-3.2-1b-it")
    
    # check_data()
    # draw()
    
    if False:
        log_path = work_dir / "checkpoints-bias/Circuit-Llama-3.2-3B-instruct-bias-epochs_3-bsz_16-lr_1e-4-Opt_SGD-top_n_3000-topn_start_0-warmup_0-cosine-loss_regularize_0.5/log.jsonl"
        ratio = []
        with jsonlines.open(log_path, "r") as f:
            for line in f:
                ratio.append(line["real_param_ratio"])
                if len(ratio) > 500:
                    break
        print(sum(ratio) / len(ratio))