"""
GMS8k
Custom dataset for Llama3.
"""

import json
import jsonlines
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import numpy as np
import os
import random


class SquadDataset(Dataset):
    
    def __init__(self, data_path, tokenizer, data_num=10000, loss_on_prefix=False):
        self.data = []
        self.tokenizer = tokenizer
        self.loss_on_prefix = loss_on_prefix
        
        with jsonlines.open(data_path, "r") as f:
            for line in f:
                paragraph = line["paragraph"]
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    answer = qa["answer"]
                    self.data.append({"question": question, "answer": answer, "context": context})

        random.shuffle(self.data)
        self.data = self.data[:data_num]
             
        for i in range(len(self.data)):
            self.data[i] = self.preprocess(self.data[i])
            
        threshold = 450
        self.data = [d for d in self.data if len(d["input_ids"]) <= threshold]
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess(self, sample):
        """
        Llama3 format.
        """
        
        context = sample["context"]
        question = sample["question"]
        answer = sample["answer"]
        
        instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease read the context and directly give the answer to the question.\n[Context]{context}\n[Question]{question}\n\nPlease directly give the final answer. If no answer can be found in the context, please output '[No Answer]' directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        response = f"The answer is:{answer}<|eot_id|>"
        
        instruction = self.tokenizer(instruction, add_special_tokens=False)
        response = self.tokenizer(response, add_special_tokens=False)
        
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

        sample = {"input_ids": input_ids, 
                  "attention_mask": attention_mask, 
                  "labels": labels,
                  "question": question,
                  "answer": answer}
        
        return sample


class SquadCollator():
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
        non_padding_keys = ["question", "answer"]
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
        return SquadDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data/train.jsonl",
                           tokenizer=tokenizer)
    else:
        raise ValueError("Invalid dataset split name.")


def get_data_collator(tokenizer):
    
    return SquadCollator(tokenizer=tokenizer)


if __name__ == "__main__":
    
    # construct_data()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SquadDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data/train.jsonl",
                            tokenizer=tokenizer)
    test_dataset = SquadDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data/dev.jsonl",
                            tokenizer=tokenizer)
    lengths = []
    for data in train_dataset:
        lengths.append(len(data["input_ids"]))
        # print(len(data["input_ids"]))
    print("Max length:", max(lengths))
    print("Min length:", min(lengths))
    print("Mean length:", np.mean(lengths))
    
    # find numbers of data that has length <= threshold
    threshold = 256
    lengths  = np.array(lengths)
    lengths = lengths <= threshold
    num = np.sum(lengths)
    print(num, 10000)
    
    print("=====================================")
    lengths = []
    for data in test_dataset:
        lengths.append(len(data["input_ids"]))
        # print(len(data["input_ids"]))
    print("Max length:", max(lengths))
    print("Min length:", min(lengths))
    print("Mean length:", np.mean(lengths))
    
    # find numbers of data that has length <= threshold
    threshold = 450
    total = len(lengths)
    lengths  = np.array(lengths)
    lengths = lengths <= threshold
    num = np.sum(lengths)
    print(num, total)
    pass
    