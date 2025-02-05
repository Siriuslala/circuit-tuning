"""
Evaluations for circuit-tuning.
"""

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import jsonlines
import re
from tqdm import tqdm
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaConfig
from llama_recipes.model_checkpointing.checkpoint_handler import load_sharded_model_single_gpu
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dist_cp

from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
from peft import get_peft_model, PeftModel
# from vllm import LLM

from eap.patching_metrics import avg_logit_diff_sv, patching_metric
from utils import load_model_from_ckpt, load_model


def evaluate_sv(model, dataloader, config, official_model_name=None, ckpt_path=None, cfg_path=None, device=None):
    """
    evaluate the performance of the model on the subject-verb agreement task.
    """
    
    if isinstance(model, str):
        model = load_model_from_ckpt(official_model_name, ckpt_path, cfg_path, device=device)
    
    ppls = []
    losses = []
    logit_diffs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["clean_inputs"]["input_ids"].to(config.device)
            attention_mask = batch["clean_inputs"]["attention_mask"].to(config.device)
            logits, loss = model(input=input_ids, attention_mask=attention_mask, return_type="both")
            
            losses.append(loss.item())
            
            # PPL        
            ppl = np.exp(loss.item())
            ppls.append(ppl)

            # logit difference
            logit_diff = avg_logit_diff_sv(logits, batch, per_prompt=False)
            logit_diffs.append(logit_diff.item())
    
    loss_final = sum(losses) / len(losses)
    ppl_final = np.exp(sum([np.log(ppl) for ppl in ppls]) / len(ppls))
    logit_diff_final = sum(logit_diffs) / len(logit_diffs)
    
    # test on spc dataset?
    # for batch in tqdm():
    #     pass
    
    ret = {"loss_final": loss_final, "ppl": ppl_final, "logit_diff": logit_diff_final}
    return ret


def simple_test(model, 
                tokenizer, 
                device, 
                prompt, 
                max_new_tokens=100, 
                return_prefix=True,
                use_vllm=False,
                use_tf_lens_tool=False):
    # if use_vllm:
    #     model = LLM("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
    #     outputs = model.generate(prompt)

    #     return [output.outputs[0].text for output in outputs]
    #     # for output in outputs:
    #     #     prompt = output.prompt # 获取原始的输入提示
    #     #     generated_text = output.outputs[0].text # 从输出对象中获取生成的文本
    #     #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    if use_tf_lens_tool:
        if isinstance(prompt, str):
            content = utils.test_prompt(model, prompt)
        else:
            raise ValueError("If you use `test_prompt` in TransformerLens, please pass in a str")
    else:
        dataloder = DataLoader(prompt, batch_size=32, shuffle=False)
        content = []
        # torch.manual_seed(14)
        
        for batch in tqdm(dataloder, desc="Evaluating on test set"):
            
            inputs = tokenizer(batch, 
                                padding=True, 
                                return_tensors='pt', 
                                padding_side="left")        
            
            # print(inputs)
            with torch.no_grad():
                if isinstance(model, HookedTransformer):
                    inputs = inputs["input_ids"].to(device)
                    outputs = model.generate(input=inputs,
                                        max_new_tokens=max_new_tokens)
                else:
                    prefix_lens = [len(input) for input in inputs["input_ids"]]
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model.generate(**inputs,
                                        max_new_tokens=max_new_tokens)
                    # print(outputs.dtype, outputs.shape)
            # print(outputs)
            # outputs = outputs.cpu().numpy()
            if return_prefix:
                content.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            else:
                for i in range(len(outputs)): 
                    content.append(tokenizer.decode(outputs[i][prefix_lens[i]:], skip_special_tokens=True))
        
    return content


def test_svd_ability(model, tokenizer, device, save_path):
    
    prompt_list = []
    sv_test_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_test/sv_test.txt"
        
    with open(sv_test_path, "r") as f:
        for line in f:
            prompt_list.append(line.strip('\n'))
    # print(prompt_list)
    content = simple_test(model, tokenizer, device, prompt_list, max_new_tokens=25)
    with jsonlines.open(save_path, "w") as f:
        for prompt, res in zip(prompt_list, content):
            line = {"prompt": prompt, "completion": res}
            f.write(line)
    
    return content



if __name__ == "__main__":
    
    def test_sv():
        
        model_name = "gpt2-small"
        
        ckpt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_30000-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/model-steps_1499_epoch_3.pt"
        # ckpt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-loss_weighted(p_1)-full_tuning/model-steps_1499_epoch_3.pt"
        
        cfg_path = "/home/lyy/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json"
        # prompt = "Every summer my friend and I"
        # prompt = ["Every summer my friend and I", "She"]  # ['Every summer my friend and I takes a shell trip to Tasmania to get a bite to eat.', "She're her own cause, after all."]
        
        
        # content = simple_test(model_name, ckpt_path, cfg_path, prompt)
        # print(content)
        
        # ckpt_path=None
        test_svd_ability(model_name, ckpt_path, cfg_path)
    
