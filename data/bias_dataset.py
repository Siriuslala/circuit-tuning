"""
GMS8k
Custom dataset for Llama3.
"""

from typing import Any
import jsonlines
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import numpy as np
    
    
class BiasDataset(Dataset):
    
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
        # threshold = 50
        # self.data = [d for d in self.data if len(tokenizer.tokenize(d["sentence"])) <= threshold]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BiasColllator:
    """
    Mofified from DataCollatorForSeq2Seq in llama recipe
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):

        prompts = [item["sentence"] for item in samples]
        pronoun_positions = [item["pronoun_pos"] for item in samples]
        pronoun_ids = [item["pronoun_id"] for item in samples]
        pronoun_anti_ids = [item["pronoun_anti_id"] for item in samples]
        
        # tokenize
        inputs = self.tokenizer.batch_encode_plus(prompts, 
                                                  padding=True,
                                                  padding_side='right', 
                                                  return_tensors='pt',
                                                  add_special_tokens=False)
        
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"].clone(),
            "pronoun_positions": pronoun_positions,
            "pronoun_ids": pronoun_ids,
            "pronoun_anti_ids": pronoun_anti_ids,
        }  # gender bias

        return batch


def get_custom_dataset(dataset_config, tokenizer, split: str):

    if split == "train":
        return BiasDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/bias/gender_bias/gender_bias_formatted_llama.jsonl",
                           tokenizer=tokenizer)
    elif split == "test":
        return None
    else:
        raise ValueError("Invalid dataset split name.")


def get_data_collator(tokenizer):
    
    return BiasColllator(tokenizer=tokenizer)


if __name__ == "__main__":
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # p1 = "I love you."
    # p2 = "You are son of a bitch"
    # prompts = [p1, p2]
    # inputs = tokenizer.batch_encode_plus(prompts, 
    #                                     padding=True,
    #                                     padding_side='right', 
    #                                     return_tensors='pt',)
    # print(inputs["input_ids"])
    # print(tokenizer.convert_ids_to_tokens([683, 607]))
    
    dataset = BiasDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/bias/gender_bias/gender_bias_formatted_llama.jsonl",
                           tokenizer=tokenizer)
    lengths = []
    for data in dataset:
        prompt = data["sentence"]
        tokens = tokenizer.tokenize(prompt)
        lengths.append(len(tokens))
    print("Max: ", max(lengths))
    print("Min: ", min(lengths))
    print("Avg: ", sum(lengths) / len(lengths))
    threshold = 70
    num = np.array(lengths) <= threshold
    print(num)
    print(sum(num))