"""
GMS8k
Custom dataset for Llama3.
"""

from typing import Any
import jsonlines
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import AutoTokenizer
import numpy as np
import re
from tqdm import tqdm

    
class MathDataset(Dataset):
    
    def __init__(self, data_path, tokenizer, data_num=None, loss_on_prefix=False):
        self.data = []
        self.tokenizer = tokenizer
        self.loss_on_prefix = loss_on_prefix
        
        cnt = 0
        with jsonlines.open(data_path, "r") as f:
            for line in f:
                self.data.append(line)
                cnt += 1
                if data_num is not None and cnt > data_num:
                    break
                
        for i in range(len(self.data)):
            self.data[i] = self.preprocess(self.data[i])
        threshold = 400
        self.data = [d for d in self.data if len(d["input_ids"]) <= threshold]
            
        # if len(self.data[i]["input_ids"]) > threshold:
        #     self.data[i]["input_ids"] = self.data[i]["input_ids"][:threshold]
        #     self.data[i]["attention_mask"] = self.data[i]["attention_mask"][:threshold]
        #     self.data[i]["labels"] = self.data[i]["labels"][:threshold]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess(self, sample):
        """
        Llama3 format.
        """
        
        question = sample["question"]
        answer = sample["answer"]
        answer_positions = sample["answer_positions"]
        answer_ids = sample["answer_ids"]
        
        instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please answer step by step and give the final answer after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        response = f"{answer}<|eot_id|>"
        
        instruction = self.tokenizer(instruction, add_special_tokens=False)
        response = self.tokenizer(response, add_special_tokens=False)
        
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

        sample = {"input_ids": input_ids, 
                  "attention_mask": attention_mask, 
                  "labels": labels,
                  "question": question,
                  "answer": answer,
                  "answer_positions": answer_positions,
                  "answer_ids": answer_ids}
        
        return sample


class MathCollator():
    """
    Mofified from DataCollatorForSeq2Seq in llama recipe
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.return_tensors = "pt"
        self.padding = True
        self.max_length = None
        self.pad_to_multiple_of = None
        self.label_pad_token_id = -100
        self.model = None

    def __call__(self, samples, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in samples[0].keys() else "labels"
        labels = [sample[label_name] for sample in samples] if label_name in samples[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_samples = [{k: v for k, v in sample.items() if k != label_name} for sample in samples]
        non_padding_keys = ["question", "answer", "answer_positions", "answer_ids"]
        padding_samples = [{k: v for k, v in sample.items() if k not in non_padding_keys} for sample in non_labels_samples]
        
        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padding_samples,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # reapply the string features
        for k in non_padding_keys:
            batch[k] = [sample[k] for sample in non_labels_samples]
        
        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False
        if labels is not None:
            if no_padding:
                if isinstance(samples[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == "max_length" and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(samples[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        return batch


def get_custom_dataset(dataset_config, tokenizer, split: str):

    if split == "train":
        return MathDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train.jsonl",
                           tokenizer=tokenizer)
    elif split == "test":
        return MathDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/test.jsonl",
                           tokenizer=tokenizer)
    else:
        raise ValueError("Invalid dataset split name.")


def get_data_collator(tokenizer):
    
    return MathCollator(tokenizer=tokenizer)


if __name__ == "__main__":
    
    def func():
        """
        Outcome guided.
        """
        tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train.jsonl"
        src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_old.jsonl"
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdi/home/lyy/models/models--meta-llama--Llama-3.2-1B-instruct")
        with jsonlines.open(src_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
            for line in f:
                data = line.copy()
                sentence_token_ids = tokenizer(line["answer"], add_special_tokens=False)["input_ids"]
                sentence_tokens = tokenizer.tokenize(line["answer"])
                sentence_tokens = [token.strip('Ġ') for token in sentence_tokens]
                start_pos = 0
                for i, token in enumerate(sentence_tokens):
                    if token == '####':
                        start_pos = i + 2
                        break
                assert start_pos != 0
                answer_positions = [i for i in range(start_pos, len(sentence_tokens))]
                answer_tokens = [sentence_tokens[i] for i in range(start_pos, len(sentence_tokens))]
                data["answer_tokens"] = answer_tokens
                data["answer_positions"] = answer_positions
                data["answer_ids"] = sentence_token_ids[start_pos:]
                f1.write(data)
            
    def func1():
        """
        No COT.
        """
        src_path = "/raid_sdi/home/lyy/lyy/Interpretability/circuit-tuning/data/gsm8k/train.jsonl"
        tgt_path = "/raid_sdi/home/lyy/lyy/Interpretability/circuit-tuning/data/gsm8k/train_new.jsonl"
        
        with jsonlines.open(src_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
            for line in f:
                data = line.copy()
                answer = line["answer"]
                for i, word in enumerate(answer):
                    if word == '#' and answer[i+1] == '#':
                        data["answer"] = answer[i:]
                        break
                f1.write(data)
    
    def func2():
        """
        Cot key words.
        """
        src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train.jsonl"
        tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_cot_analyses.jsonl"
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct")
        with jsonlines.open(src_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
            lines = list(f)
            for line in tqdm(lines):
                data = line.copy()
                question = line["question"]
                answer = line["answer"]
                
                instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please answer step by step and give the final answer after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                response = f"{answer}<|eot_id|>"
                
                instruction_tokens = tokenizer.tokenize(instruction, add_special_tokens=False)
                response_tokens = tokenizer.tokenize(response, add_special_tokens=False)
                instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                # print(response_ids)
                # print(instruction)
                # print(instruction_tokens)
                # print(response)
                # print(response_tokens)
                cr_id = tokenizer('\n', add_special_tokens=False)["input_ids"][0]
                double_cr_id = tokenizer('\n\n', add_special_tokens=False)["input_ids"][0]
                stop_with_cr_id = tokenizer('.\n', add_special_tokens=False)["input_ids"][0]
                # print(tokenizer.convert_ids_to_tokens([cr_id, double_cr_id, stop_with_cr_id]))
                # print(cr_id, double_cr_id, stop_with_cr_id)
                
                cot_key_words_pos = [len(instruction_tokens) - 1]
                step_key_words_pos = []
                for i, (token_id, token) in enumerate(zip(response_ids, response_tokens)):
                    # if token_id == stop_with_cr_id or token_id == cr_id:  # step.\n or step\n
                    if 'Ċ' in token:
                        step_key_words_pos.append(i + len(instruction_ids))
                        
                # check
                input_ids = instruction_ids + response_ids
                assert input_ids[cot_key_words_pos[0]] == double_cr_id
                for pos in step_key_words_pos:
                    if not '\n' in tokenizer.decode([input_ids[pos]]):
                        print(response)
                        print(response_tokens)
                        print(tokenizer.decode([input_ids[pos]]))
                        print("====="*10)

                if not step_key_words_pos:
                    print(response)
                    print(response_tokens)
                    import pdb; pdb.set_trace()
                else:
                    data["cot_key_words_pos"] = cot_key_words_pos
                    data["step_key_words_pos"] = step_key_words_pos
                    f1.write(data)
    
    def func3():
        """
        DA key words.
        """
        src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_new.jsonl"
        tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_da_analyses.jsonl"
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct")
        with jsonlines.open(src_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
            lines = list(f)
            for line in tqdm(lines):
                data = line.copy()
                question = line["question"]
                answer = line["answer"]
                
                instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please directly answer the question with final result after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                response = f"{answer}<|eot_id|>"
                
                instruction_tokens = tokenizer.tokenize(instruction, add_special_tokens=False)
                response_tokens = tokenizer.tokenize(response, add_special_tokens=False)
                instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                # print(response_ids)
                # print(instruction)
                # print(instruction_tokens)
                # print(response)
                # print(response_tokens)
                cr_id = tokenizer('\n', add_special_tokens=False)["input_ids"][0]
                double_cr_id = tokenizer('\n\n', add_special_tokens=False)["input_ids"][0]
                stop_with_cr_id = tokenizer('.\n', add_special_tokens=False)["input_ids"][0]
                # print(tokenizer.convert_ids_to_tokens([cr_id, double_cr_id, stop_with_cr_id]))
                # print(cr_id, double_cr_id, stop_with_cr_id)
                
                cot_key_words_pos = [len(instruction_tokens) - 1]
                        
                data["da_key_words_pos"] = cot_key_words_pos
                f1.write(data)
    
    def func4(mode="cot"):
        """
        get "Please answer step by step" key words or "Please directly answer" key words.
        """ 
        
        if mode == "cot":
            src_path = "//raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_cot_analyses.jsonl"
            tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_cot_analyses_new.jsonl"
        else:
            src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_da_analyses.jsonl"
            tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_da_analyses_new.jsonl"
            
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct")
        with jsonlines.open(src_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
            lines = list(f)
            for line in tqdm(lines):
                data = line.copy()
                question = line["question"]
                answer = line["answer"]
                
                if mode == "cot":
                    instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please answer step by step and give the final answer after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                else:
                    
                    instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question} Please directly answer the question with final result after '#### '.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                response = f"{answer}<|eot_id|>"
                
                instruction_tokens = tokenizer.tokenize(instruction, add_special_tokens=False)
                instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)

                # print(instruction_tokens)
                start_id = 0
                for i, token in enumerate(instruction_tokens):
                    if mode == "cot":
                        if token == 'ĠPlease' and instruction_tokens[i+1] == 'Ġanswer':
                            start_id = i
                            break
                    else:
                        if token == 'ĠPlease' and instruction_tokens[i+1] == 'Ġdirectly':
                            start_id = i
                            break
                if mode == "cot":
                    data["step_by_step_key_words_pos"] = [start_id + i for i in range(5)][1:]
                else:
                    data["directly_ans_key_words_pos"] = [start_id + i for i in range(3)][1:]
                f1.write(data)

    def func5():
        """
        No instruction.
        """ 
        
        src_path = "//raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train.jsonl"
        tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gsm8k/train_analyses_no_it.jsonl"
            
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct")
        with jsonlines.open(src_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
            lines = list(f)
            for line in tqdm(lines):
                data = line.copy()
                del data["answer_tokens"]
                del data["answer_positions"]
                del data["answer_ids"]
                question = line["question"]
                
                instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                instruction_tokens = tokenizer.tokenize(instruction, add_special_tokens=False)
                instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)

                # print(instruction_tokens)
                data["double_cr_id"] = [len(instruction_tokens) - 1]
                print(instruction_tokens[data["double_cr_id"][0]])
                f1.write(data)
    
    # func1()
    # func3()
    
    func5()