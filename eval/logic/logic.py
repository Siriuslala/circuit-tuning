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
            test_data.append(line)
    
    instruction = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    questions = [instruction.format(question=sample["question"]) for sample in test_data]
    answers = [sample["answer"] for sample in test_data]
    print(questions[0])
    print(answers[0])
        
    return questions, answers

def generate(model, 
            tokenizer, 
            device, 
            prompts, 
            max_new_tokens=400, 
            return_prefix=False):
  
    dataloader = DataLoader(prompts, batch_size=16, shuffle=False)
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

def parse_result(result):
    answer = 'no answer'
    try:
        answer = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[-1]
    except:
        answer_suffix = result[:-len("The correct answer is N/A (undetermined).")].lower()
        if "true" in answer_suffix:
            answer = "true"
        elif "false" in answer_suffix:
            answer = "false"
        elif "n/a" in answer_suffix:
            answer = "n/a"
        else:
            answer = "no answer"
    return answer

def clean_number(n):
    n = n.replace('$', '').replace(',', '')
    return n

def check_correctness(output, answer):
    try: 
        output = clean_number(output)
        if output == answer:
            #print('correct', (output, answer))
            return True
        output = float(output)
        if output == answer:
            #print('correct', (output, answer))
            return True
    except:
        if output in ['true', 'false', 'n/a']:
            if output == answer or bool(output) == answer:
                #print('correct', (output, answer))
                return True
    #print('wrong', (output, answer))
    return False

def compute_PRF(results):
    true_TP = len([a for a in results if check_correctness(a[0], a[1]) and a[0] == 'true'])
    true_FP = len([a for a in results if a[0] == 'true' and a[1] != 'true'])
    true_TN = len([a for a in results if check_correctness(a[0], a[1]) and a[1] != 'true'])
    true_FN = len([a for a in results if a[0] != 'true' and a[1] == 'true'])
    if (true_TP + true_FP) != 0:
        true_precision = true_TP/(true_TP + true_FP)
    else:
        true_precision = None
    if (true_TP + true_FN) != 0:
        true_recall = true_TP/(true_TP + true_FN)
    else:
        true_recall = None
    true_f1 = None
    if true_recall is not None:
        if true_precision is not None:
            if true_precision + true_recall != 0:
                true_f1 = 2*true_precision*true_recall/(true_precision + true_recall)
            else:
                true_f1 = 0
        else:
            true_f1 = 0
    
    false_TP = len([a for a in results if check_correctness(a[0], a[1]) and a[0] == 'false'])
    false_FP = len([a for a in results if a[0] == 'false' and a[1] != 'false'])
    false_TN = len([a for a in results if check_correctness(a[0], a[1]) and a[1] != 'false'])
    false_FN = len([a for a in results if a[0] != 'false' and a[1] == 'false'])

    if (false_TP + false_FP) != 0:
        false_precision = false_TP/(false_TP + false_FP)
    else:
        false_precision = None
    if (false_TP + false_FN) != 0:
        false_recall = false_TP/(false_TP + false_FN)
    else:
        false_recall = None
    false_f1 = None
    if false_recall is not None:
        if false_precision is not None:
            if false_precision + false_recall != 0:
                false_f1 = 2*false_precision*false_recall/(false_precision + false_recall)
            else:
                false_f1 = 0
        else:
            false_f1 = 0

    na_TP = len([a for a in results if check_correctness(a[0], a[1]) and a[0] == 'n/a'])
    na_FP = len([a for a in results if a[0] == 'n/a' and a[1] != 'n/a'])
    na_TN = len([a for a in results if check_correctness(a[0], a[1]) and a[1] != 'n/a'])
    na_FN = len([a for a in results if a[0] != 'n/a' and a[1] == 'n/a'])

    if (na_TP + na_FP) != 0:
        na_precision = na_TP/(na_TP + na_FP)
    else:
        na_precision = None
    if (na_TP + na_FN) != 0:
        na_recall = na_TP/(na_TP + na_FN)
    else:
        na_recall = None
    na_f1 = None
    if na_recall is not None:
        if na_precision is not None:
            if na_precision + na_recall != 0:
                na_f1 = 2*na_precision*na_recall/(na_precision + na_recall)
            else:
                na_f1 = 0
        else:
            na_f1 = 0

    average_precision = np.mean([a for a in [true_precision, false_precision, na_precision] if a is not None])
    average_recall = np.mean([a for a in [true_recall, false_recall, na_recall] if a is not None])
    average_f1 = np.mean([a for a in [true_f1, false_f1, na_f1] if a is not None])
    return average_precision, average_recall, average_f1, {
        'true': [true_precision, true_recall, true_f1], 
        'false': [false_precision, false_recall, false_f1], 
        'n/a': [na_precision, na_recall, na_f1]
        }

def present_PRF(results):
    print('=====================')
    average_precision, average_recall, average_f1, PRF = compute_PRF(results)
    print(f'average precision: {average_precision}')
    print(f'average recall: {average_recall}')
    print(f'average f1: {average_f1}')

def main(args):
    
    model, tokenizer = load_model(args.model_name, args.ckpt_path, args.device)
    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic/logic_test.json"
    output_dir = os.path.join(args.save_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    questions, answers = gen_prompts(data_path)
    responses = generate(model, tokenizer, args.device, questions)
    result_path = os.path.join(output_dir, "results.json")
    with jsonlines.open(result_path, "w") as writer:
        for i in range(len(questions)):
            writer.write({"question": questions[i], "answer": answers[i], "response": responses[i]})
    
    results_for_checking = []
    for ans, res in zip(answers, responses):
        ans = parse_result(ans)
        model_ans = parse_result(res)
        results_for_checking.append((model_ans.lower(), ans.lower()))
    present_PRF(results_for_checking)
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/logic/results")
    parser.add_argument("--model_name", type=str, default="llama-3.2-1b-it")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--output_dir", type=str, default="llama-3.2-1b-it")
    args = parser.parse_args()
    
    main(args)
