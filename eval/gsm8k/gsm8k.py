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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_model_from_ckpt, hookdedTF_to_TF

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE_1 = re.compile(r"boxed\{(\-?[0-9\.\,]+)\}")  # boxed{64}
ANS_RE_2 = re.compile(r"The final answer is: \$?(\-?[0-9\.\,]+)\$?(\[Answer\]|\.)")  # The final answer is: $26$[Answer]          The final answer is: $57.00.
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    match1 = ANS_RE_1.search(completion)
    match2 = ANS_RE_2.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "").replace("$", "").replace(".00", "")
        return match_str
    elif match1:
        match_str = match1.group(1).strip()
        match_str = match_str.replace(",", "").replace("$", "").replace(".00", "")
        return match_str
    elif match2:
        match_str = match2.group(1).strip()
        match_str = match_str.replace(",", "").replace("$", "").replace(".00", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS
    exact_correct = extract_answer(model_completion) == gt_answer
    completion_tail = model_completion[-20:].replace(",", "")
    possible_ans = [' ' + gt_answer,
                    '{'+ gt_answer + '}',
                    '$' + gt_answer ]
    rough_correct = any([ans in completion_tail for ans in possible_ans])
    
    return exact_correct or rough_correct

def evaluate_math(model, dataloader, config, official_model_name=None, ckpt_path=None, cfg_path=None, device=None):
    """
    evaluate the performance of the model on the math task.
    """
    
    if isinstance(model, str):
        model = load_model_from_ckpt(official_model_name, ckpt_path, cfg_path, device=device)
    
    ppls = []
    losses = []
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            loss = model(input=input_ids, attention_mask=attention_mask, return_type="loss")
            losses.append(loss.item())       
            ppl = np.exp(loss.item())
            ppls.append(ppl)
            
            inputs = model.tokenizer.batch_encode_plus(batch["questions"], 
                                                    padding=True, 
                                                    return_tensors='pt', 
                                                    padding_side="left")["input_ids"]
            # print(inputs)
            with torch.no_grad():
                outputs = model.generate(input=inputs,
                                        max_new_tokens=400)
            answers = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for ans, gt in zip(answers, batch["answers"]):
                if is_correct(ans, gt):
                    correct += 1
                total += 1    

    
    loss_final = sum(losses) / len(losses)
    ppl_final = np.exp(sum([np.log(ppl) for ppl in ppls]) / len(ppls))
    acc = correct / total
    
    ret = {"loss_final": loss_final, "ppl": ppl_final, "acc@1": acc}
    return ret

def evaluate_math_llama_recipe(model, tokenizer, dataloader, device):
    """
    Evaluate the performance of the model on the math task.
    Used for the model trained with the llama recipe.
    """
    
    total = 0
    correct_num = 0
    test_results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for key in batch.keys():
                if not key in ["input_ids", "attention_mask", "labels"]:
                    continue
                batch[key] = batch[key].to(device)
            # remove answer
            mask = batch["labels"] == -100
            batch["input_ids"] = batch["input_ids"].masked_fill(~mask, tokenizer.pad_token_id)
            for i in range(len(batch)):
                qn_end = torch.nonzero(batch["labels"][i] == -100)[-1]
                batch["input_ids"][i] = torch.cat([batch["input_ids"][i][qn_end+1:], batch["input_ids"][i][:qn_end+1]])
                
            outputs = model.generate(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"], 
                                    max_length=512)
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for qn, gt, ans in zip(batch["question"], batch["answer"], answers):
                correct = is_correct(ans, gt)
                if correct:
                    correct_num += 1
                total += 1    
                test_results.append({"question": qn,
                                     "answer": gt,
                                     "model_ans": ans,
                                     "is_correct": correct})

    acc = correct_num / total   
    test_metrics = {"acc@1": acc}
    test_results.append(test_metrics)
    
    return test_metrics, test_results

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
        dataloader = DataLoader(prompt, batch_size=16, shuffle=False)
        content = []
        # torch.manual_seed(14)
        
        for batch in tqdm(dataloader, desc="Evaluating on test set"):
            
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
                                             max_new_tokens=max_new_tokens,
                                             )
                    # print(outputs.dtype, outputs.shape)
            # print(outputs)
            # outputs = outputs.cpu().numpy()
            if return_prefix:
                content.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            else:
                for i in range(len(outputs)): 
                    content.append(tokenizer.decode(outputs[i][prefix_lens[i]:], skip_special_tokens=True))
        
    return content

def test_math_ability_original(model, tokenizer, device, cot, save_path):
    
    test_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/math_dataset/test.jsonl"
    
    test_data = []
    with jsonlines.open(test_path, "r") as f:
        for line in f:
            test_data.append(line)
    test_data = test_data[:]
    
    # if cot:
    #     prompt = [
    #         "[Question]Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [Answer]\nNatalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n\#### 72",
    #         "[Question]Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? [Answer]\nWeng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"]
    #     prompt = '\n\n'.join(prompt) + '\n\n'
    #     questions = [prompt + "[Question]" + sample["question"] + "[Answer]\n" for sample in test_data]
    # else:
    #     questions = ["[Question]" + sample["question"] + "[Answer]\n" for sample in test_data]
    
    cot_prompt = "[Question]Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [Answer]\nNatalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n\#### 72\n\n[Question]Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? [Answer]\nWeng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"
    instruction = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please answer step by step and give the final answer after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    if cot:
        questions = [instruction.format(question=cot_prompt + "\n\n" + sample["question"] + "\n\n") for sample in test_data]
    else:
        questions = [instruction.format(question=sample["question"]) for sample in test_data]
    answers = [sample["answer"] for sample in test_data]
    answers = [sample["answer"] for sample in test_data]
    
    print(questions[0])
    content = simple_test(model, 
                          tokenizer, 
                          device, 
                          questions, 
                          max_new_tokens=400, 
                          return_prefix=False,
                          use_vllm=False)
    with jsonlines.open(save_path, "w") as f:
        for qn, ans, res in zip(questions, answers, content):
            ans_is_correct = is_correct(res, ans)
            line = {"question": qn, "answer": ans, "model_ans": res, "is_correct": ans_is_correct}
            f.write(line)
    
    return content


def test_math_ability(model, tokenizer, device, cot, save_path):
    
    test_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/test.jsonl"
    
    save_dir = '/'.join(save_path.split("/")[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    test_data = []
    with jsonlines.open(test_path, "r") as f:
        for line in f:
            test_data.append(line)
    test_data = test_data[:]
    
    cot_prompt = "[Question]Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [Answer]\nNatalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n\#### 72\n\n[Question]Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? [Answer]\nWeng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"
    instruction = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please answer step by step and give the final answer after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    if cot:
        questions = [instruction.format(question=cot_prompt + "\n\n" + sample["question"] + "\n\n") for sample in test_data]
    else:
        questions = [instruction.format(question=sample["question"]) for sample in test_data]
    answers = [sample["answer"] for sample in test_data]
    
    print(questions[0])
    content = simple_test(model, 
                          tokenizer, 
                          device, 
                          questions, 
                          max_new_tokens=400, 
                          return_prefix=False,
                          use_vllm=False)
    correct = 0
    with jsonlines.open(save_path, "w") as f:
        for qn, ans, res in zip(questions, answers, content):
            ans_is_correct = is_correct(res, ans)
            correct += ans_is_correct
            line = {"question": qn, "answer": ans, "model_ans": res, "is_correct": ans_is_correct}
            f.write(line)
    print(f"result: {correct}/{len(questions)}")
    
    return content


if __name__ == "__main__":
    
    def test_math(model_name, ckpt_path=False, device="cuda:3", cot=False, save_path=None):
        
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
                                                        torch_dtype=torch.bfloat16).eval()
                model = PeftModel.from_pretrained(
                    model=hf_model,
                    model_id=ckpt_path,
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
                    # torch.distributed.init_process_group("nccl")
                    # rank = int(os.environ["RANK"])
                    rank=0
                    device_id = torch.cuda.current_device()
                    
                    from llama_recipes.configs import fsdp_config as FSDP_CONFIG
                    from llama_recipes.utils import fsdp_auto_wrap_policy
                    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                    from llama_recipes.utils.train_utils import get_policies
                    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
                    from llama_recipes.utils.fsdp_utils import hsdp_device_mesh

                    
                    fsdp_config = FSDP_CONFIG()
                    fsdp_config.pure_bf16 = True
                    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
                    mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank=0)

                    # model = FSDP(
                    #     model,
                    #     auto_wrap_policy=(
                    #         wrapping_policy
                    #     ),
                    #     cpu_offload=(
                    #         CPUOffload(offload_params=True)
                    #         if fsdp_config.fsdp_cpu_offload
                    #         else None
                    #     ),
                    #     mixed_precision=(
                    #         mixed_precision_policy if not fsdp_config.pure_bf16 else None
                    #     ),
                    #     sharding_strategy=fsdp_config.sharding_strategy,
                    #     device_mesh=None,
                    #     device_id=device_id,
                    #     limit_all_gathers=True,
                    #     sync_module_states=False,
                    #     param_init_fn=None,
                    # )
                    # model = load_sharded_model_single_gpu(model, ckpt_path)
                    
                    if rank == 0:
                        print(f"loading model from model path: {ckpt_path} ")
                    state_dict = {
                        "model": model.state_dict()
                    }
                    dist_cp.load(state_dict=state_dict,
                                         checkpoint_id=ckpt_path)
                    model.load_state_dict(state_dict["model"])
                    
                    # reader = torch.distributed.checkpoint.FileSystemReader(ckpt_path)

                    # with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    #     checkpoint = {"model": model.state_dict()}
                    #     if rank == 0:
                    #         ck = checkpoint.keys()
                    #         print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
                    
                    #     dist_cp.load(
                    #         state_dict=checkpoint,
                    #         storage_reader=reader,
                    #     )
                    #     if rank == 0:
                    #         print(f"checkpoint after load_state_dict()")
                    #         ck = checkpoint.keys()
                    #         print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
                    #     model.load_state_dict(checkpoint["model"])
                    # if rank == 0:
                    #     print(f"Sharded state checkpoint loaded from {ckpt_path}")
                # print(type(model))
                # import pdb; pdb.set_trace()
                model.eval().bfloat16()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                        torch_dtype=torch.bfloat16)
            # print(type(model))
            # import pdb; pdb.set_trace()
            model.eval().bfloat16()
        
        model.to(device)
        if ckpt_path:
            test_math_ability(model, tokenizer, device, cot, save_path)
        else:
            test_math_ability_original(model, tokenizer, device, cot, save_path)
    
    def fix_math_result(original_path, new_path):
        with jsonlines.open(original_path, 'r') as f, jsonlines.open(new_path, 'w') as f1:
            for line in f:
                model_ans = line["model_ans"]
                gt = line["answer"]
                correct = is_correct(model_ans, gt)
                new_line = {k: v for k, v in line.items()}
                new_line["is_correct"] = correct
                f1.write(new_line)
    
    # ===================== circuit-tuning
    test_math(model_name="llama-3.2-3b-it",
    ckpt_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/Circuit-Llama-3.2-3B-instruct-math-epochs_3-bsz_16-lr_5e-5-Opt_SGD-top_n_3000-topn_start_0-warmup_0-cosine-guide_process-ablation_mean-ie_over_seq-loss_weighted_p_1-prune_top_nodes/fsdp-epoch_500steps",
            device="cuda:1",
            cot=False,
            save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/gsm8k/results/llama3.2-3b-circuit-lr3e-5-circuit-300steps/output.jsonl")
    
    # ===================== lora
    # test_math(model_name="llama-3.2tmux -1b-it",
    # ckpt_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/llama-3.2-1B-it-math-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16/peft_1",
    #         device="cuda:2",
    #         cot=False,
    #         save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.2-1b/math_test_llama3.2-1b-it_eos_pad-epochs_500steps-bsz_16-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16.jsonl")
    
    # test_math(model_name="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/llama-3.2-3B-it-math-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16/peft_1",
    #           device="cuda:0",
    #           cot=False,
    #           save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.2-3b/math_test_llama3.2-3b-it_eos_pad-epochs_500steps-bsz_16-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16_again.jsonl")
    
    # test_math(model_name="llama-3.1-8b-it",
    #     ckpt_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/llama-3.1-8B-it-math-epochs_3-bsz_128-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16",
    #           device="cuda:2",
    #           cot=False,
    #           save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.1-8b/math_test_llama3.1-8b-it_eos_pad-epochs_3-bsz_128-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16_again.jsonl")
    
    
    # ===================== full
    # test_math(model_name="llama-3.2-1b-it",
    #     ckpt_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/llama-3.2-1B-it-math-epochs_1-bsz_96-lr_3e-06-Opt_SGD-full-precision_bf16-1/fsdp-/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct",
    #         device="cuda:1",
    #         cot=False,
    #         save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.2-1b/math_test_llama3.2-1b-it_eos_pad-epochs_3-bsz_96-lr_3e-6-Opt_SGD-full-precision_bf16_train.jsonl")
    
    # test_math(model_name="llama-3.2-3b-it",
    #     ckpt_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/llama-3.2-3B-it-math-epochs_1-bsz_96-lr_3e-06-Opt_SGD-full-precision_bf16-1/fsdp-/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-3B-instruct",
    #         device="cuda:1",
    #         cot=False,
    #         save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.2-3b/math_test_llama3.2-3b-it_eos_pad-epochs_1-bsz_96-lr_3e-6-Opt_SGD-full-precision_bf16.jsonl")
    
    # test_math(model_name="llama-3.1-8b-it",
    # ckpt_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/llama-3.1-8B-it-math-epochs_1-bsz_64-lr_1e-06-Opt_SGD-full-precision_bf16-1/fsdp-/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct",
    #     device="cuda:2",
    #     cot=False,
    #     save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.1-8b/math_test_llama3.1-8b-it_eos_pad-epochs_1-bsz_64-lr_3e-6-Opt_SGD-full-precision_bf16.jsonl")
   
   # ====================== original
    # test_math(model_name="llama-3.1-8b-it",
    #           ckpt_path=None,
    #           device="cuda:0",
    #           cot=False,
    #           save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.1-8b/math_test_llama3.1-8b-it_eos_pad_no_cot_again_again_again.jsonl")
    
    # test_math(model_name="llama-3.2-3b-it",
    #         ckpt_path=None,
    #         device="cuda:1",
    #         cot=False,
    #         save_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eval/math_test/llama-3.2-3b/math_test_llama3.2-3b-it_eos_pad_no_cot_again_again.jsonl")

    
    
    # fix_math_result(original_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/math_test/math_test_llama3.2-3b-it_eos_pad-epochs_3-bsz_128-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16.jsonl",
    #                 new_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/math_test/llama-3.2-3b/fix_math_test_llama3.2-3b-it_eos_pad-epochs_3-bsz_128-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16.jsonl")
    
    