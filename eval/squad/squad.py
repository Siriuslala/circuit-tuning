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
from peft import get_peft_model, PeftModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_model_from_ckpt, hookdedTF_to_TF

import string
import collections
import argparse


def load_model(model_name, ckpt_path, device):
    if "3b" in model_name.lower():
        model_name = "meta-llama/Llama-3.2-3B-instruct"
        model_path = "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-3B-instruct"
    elif "1b" in model_name.lower():
        model_name = "meta-llama/Llama-3.2-1B-instruct"
        model_path = "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct"
    elif "8b" in model_name.lower():
        model_name = "meta-llama/Llama-3.1-8B-instruct"
        model_path = "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct"
    else:
        pass
    
    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    if ckpt_path:  # peft or full-tuning
        if "lora" in ckpt_path:
            hf_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    torch_dtype=torch.bfloat16).eval().to(device)
            model = PeftModel.from_pretrained(
                model=hf_model,
                model_id=ckpt_path,
                torch_device=device,
            )
        elif "Circuit" in ckpt_path:
            config = LlamaConfig.from_pretrained(model_path)
            cfg_path = os.path.join(model_path, "config.json")
            model = load_model_from_ckpt(model_name, ckpt_path, cfg_path, tokenizer=tokenizer, move_to_device=False)
            model = hookdedTF_to_TF(model_name, model, device=device)
            model.eval().bfloat16()
        else:
            model = LlamaForCausalLM(config)
            if "fsdp" in ckpt_path.lower():
                rank=0
                if rank == 0:
                    print(f"loading model from model path: {ckpt_path} ")
                state_dict = {
                    "model": model.state_dict()
                }
                dist_cp.load(state_dict=state_dict,
                            checkpoint_id=ckpt_path)
                model.load_state_dict(state_dict["model"])
            
            model.eval().bfloat16()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    torch_dtype=torch.bfloat16).eval()

    model.to(device)
        
    return model, tokenizer

def gen_prompts(test_path):
    
    test_data = []
    with jsonlines.open(test_path, "r") as f:
        for line in f:
            paragraph = line["paragraph"]
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                id = qa["id"]
                question = qa["question"]
                answer = qa["answer"]
                test_data.append({"id": id, "question": question, "answer": answer, "context": context})
    
    # test_data = test_data[:10]             
    instruction = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease read the context and directly give the answer to the question.\n[Context]{context}\n[Question]{question}\n\nPlease directly give the final answer. If no answer can be found in the context, please output '[No Answer]' directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:"

    ids = [sample["id"] for sample in test_data]
    questions = [instruction.format(context=sample["context"], question=sample["question"]) for sample in test_data]
    answers = [sample["answer"] for sample in test_data]
    
    print(ids[0])
    print(questions[0])
    print(answers[0])
        
    return ids, questions, answers

def generate(model, 
            tokenizer, 
            device, 
            prompts, 
            max_new_tokens=400, 
            return_prefix=False):
  
    dataloader = DataLoader(prompts, batch_size=32, shuffle=False)
    responses = []
    # torch.manual_seed(14)
    
    for batch in tqdm(dataloader, desc="Evaluating on test set"):
        
        inputs = tokenizer(batch, 
                            padding=True, 
                            return_tensors='pt', 
                            padding_side="left")        
        
        with torch.no_grad():
            prefix_lens = [len(input) for input in inputs["input_ids"]]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs,
                                     max_new_tokens=max_new_tokens)
        if return_prefix:
            responses.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        else:
            for i in range(len(outputs)): 
                responses.append(tokenizer.decode(outputs[i][prefix_lens[i]:], skip_special_tokens=True))
    return responses

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def parse_result(result):
    if "no answer" in result.lower():
        return ''
    else:
        return result

def main(args):
    
    model, tokenizer = load_model(args.model_name, args.ckpt_path, args.device)
    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data/dev.jsonl"
    output_dir = os.path.join(args.save_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    ids, questions, answers = gen_prompts(data_path)
    responses = generate(model, tokenizer, args.device, questions)
    result_path = os.path.join(output_dir, "results.jsonl")
    with jsonlines.open(result_path, "w") as writer:
        for i in range(len(questions)):
            writer.write({"id": ids[i], "question": questions[i], "answer": answers[i], "response": responses[i]})
    
    exact_scores = []
    f1_scores = []
    for ans_list, res in zip(answers, responses):
        ans_list = [parse_result(ans) for ans in ans_list]
        model_ans = parse_result(res)
        exact_scores.append(max([compute_exact(ans, model_ans) for ans in ans_list]))
        f1_scores.append(max([compute_f1(ans, model_ans) for ans in ans_list]))
    print(f"Exact match: {np.mean(exact_scores)}")
    print(f"F1 score: {np.mean(f1_scores)}")
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/squad/results")
    parser.add_argument("--model_name", type=str, default="llama-3.2-1b-it")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--output_dir", type=str, default="llama-3.2-1b-it")
    args = parser.parse_args()
    
    main(args)


    # exact_scores = []
    # f1_scores = []
    # with jsonlines.open("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/squad/results/llama-3.2-1b-it/results.jsonl", "r") as f:
    #     for line in f:
    #         ans = line["answer"]
    #         model_ans = line["response"]
    #         print(ans)
    #         print(model_ans)
    #         ans_list = [parse_result(ans) for ans in ans]
    #         model_ans = parse_result(model_ans)
    #         print(ans_list)
    #         print(model_ans)
    #         print([compute_exact(ans, model_ans) for ans in ans_list])
    #         print([compute_f1(ans, model_ans) for ans in ans_list])
    #         exact_scores.append(max([compute_exact(ans, model_ans) for ans in ans_list]))
    #         f1_scores.append(max([compute_f1(ans, model_ans) for ans in ans_list]))
    #         print("===")
    #     print("Exact match: ", np.mean(exact_scores))
    #     print("F1 score: ", np.mean(f1_scores))
