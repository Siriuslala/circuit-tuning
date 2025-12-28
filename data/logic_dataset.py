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

EVALUATE_PROMPTS = {
    "deductive_logic":
"""
Analyze and answer the deductive logical problem below step by step.

Deductive logic problem:
{}

Answer the question use: <answer>True</answer> or <answer>False</answer> or <answer>N/A</answer>, N/A means the result can not be deduced.
""",

    "abductive_logic":
"""
Analyze and answer the abductive logic problem below step by step.

Abductive logic problem:
{}

Answer the question use: <answer>True</answer> or <answer>False</answer> or <answer>N/A</answer>, N/A means the result can not be abduced.
""",
}

def load_data(data_dir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic", levels=[1, 2]):
    def collect_data(datas, category):
        collected_data = []
        for data, reasoning_type in datas:
            for template in data:
                for domain in template.keys():
                    if domain == category:
                        for subcat, subdata in template[domain].items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[reasoning_type].format(text)
                            answer = '<answer>'+ str(template['answer']).capitalize() +'</answer>'
                            cot = subdata['gold_cot']
                            collected_data.append({'question': text, 'answer': answer, 'cot': cot})
        return collected_data
    
    with open(os.path.join(data_dir, 'data_level1/abductive_logic_traincot.json'), 'r') as f:
        abductive_logic_level1 = json.load(f)
    with open(os.path.join(data_dir, 'data_level2/abductive_logic_traincot.json'), 'r') as f:
        abductive_logic_level2 = json.load(f)
    with open(os.path.join(data_dir, 'data_level3/abductive_logic_traincot.json'), 'r') as f:
        abductive_logic_level3 = json.load(f)
    with open(os.path.join(data_dir, 'data_level4/abductive_logic_traincot.json'), 'r') as f:
        abductive_logic_level4 = json.load(f)
    
    with open(os.path.join(data_dir, 'data_level1/deductive_logic_traincot.json'), 'r') as f:
        deductive_logic_level1 = json.load(f)
    with open(os.path.join(data_dir, 'data_level2/deductive_logic_traincot.json'), 'r') as f:
        deductive_logic_level2 = json.load(f)
    with open(os.path.join(data_dir, 'data_level3/deductive_logic_traincot.json'), 'r') as f:
        deductive_logic_level3 = json.load(f)
    with open(os.path.join(data_dir, 'data_level4/deductive_logic_traincot.json'), 'r') as f:
        deductive_logic_level4 = json.load(f)

    # datas = [(abductive_logic_level1, 'abductive_logic'), (abductive_logic_level2, 'abductive_logic'), (abductive_logic_level3, 'abductive_logic'), (abductive_logic_level4, 'abductive_logic'), (deductive_logic_level1, 'deductive_logic'), (deductive_logic_level2, 'deductive_logic'), (deductive_logic_level3, 'deductive_logic'), (deductive_logic_level4, 'deductive_logic')]
    
    data_level1 = [(abductive_logic_level1, 'abductive_logic'), (deductive_logic_level1, 'deductive_logic')]
    data_level2 = [(abductive_logic_level2, 'abductive_logic'), (deductive_logic_level2, 'deductive_logic')]
    data_level3 = [(abductive_logic_level3, 'abductive_logic'), (deductive_logic_level3, 'deductive_logic')]
    data_level4 = [(abductive_logic_level4, 'abductive_logic'), (deductive_logic_level4, 'deductive_logic')]
    data_level_map = {1: data_level1, 2: data_level2, 3: data_level3, 4: data_level4}
    datas = []
    for level in levels:
        datas += data_level_map[level]

    culture_data = collect_data(datas, 'culture and arts')
    geography_data = collect_data(datas, 'geography and places')
    activity_data = collect_data(datas, 'human activities')
    math_data = collect_data(datas, 'mathematics and logic')
    science_data = collect_data(datas, 'natural and physical sciences')
    people_data = collect_data(datas, 'people and self')
    philosophy_data = collect_data(datas, 'philosophy and thinking')
    religion_data = collect_data(datas, 'religion and belief systems')
    society_data = collect_data(datas, 'society and social sciences')
    technology_data = collect_data(datas, 'technology and applied sciences')
    health_data = collect_data(datas, 'health and fitness')
    abstract_data = collect_data(datas, 'abstract')

    return culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data

def construct_data(data_dir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic"):
    culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data = load_data()
    all_data = culture_data + geography_data + activity_data + math_data + science_data + people_data + philosophy_data + religion_data + society_data + technology_data + health_data + abstract_data
    
    random.shuffle(all_data)
    test_data = all_data[:200]
    train_data = all_data[200:]
    train_data_path = os.path.join(data_dir, 'logic_train.json')
    test_data_path = os.path.join(data_dir, 'logic_test.json')
    with jsonlines.open(train_data_path, "w") as f:
        for data in train_data:
            f.write(data)
    with jsonlines.open(test_data_path, "w") as f:
        for data in test_data:
            f.write(data)
    
    # all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))
    print('Data size:', len(all_data))
    print('Example:', all_data[0])

    
class LogicDataset(Dataset):
    
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
        
        question = sample["question"]
        answer = sample["cot"]
        
        instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
                  "answer": answer}
        
        return sample


class LogicCollator():
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
        return LogicDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic/logic_train.json",
                           tokenizer=tokenizer)
    elif split == "test":
        return LogicDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic/logic_test.json",
                           tokenizer=tokenizer)
    else:
        raise ValueError("Invalid dataset split name.")


def get_data_collator(tokenizer):
    
    return LogicCollator(tokenizer=tokenizer)


if __name__ == "__main__":
    
    # construct_data()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = LogicDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic/logic_train.json",
                            tokenizer=tokenizer)
    test_dataset = LogicDataset(data_path="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic/logic_test.json",
                            tokenizer=tokenizer)
    lengths = []
    for data in train_dataset:
        lengths.append(len(data["input_ids"]))
        # print(len(data["input_ids"]))
    print("Max length:", max(lengths))
    print("Min length:", min(lengths))
    print("Mean length:", np.mean(lengths))
    
    # find numbers of data that has length <= threshold
    threshold = 450
    lengths  = np.array(lengths)
    lengths = lengths <= threshold
    num = np.sum(lengths)
    print(num, 5560)
    
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
    lengths  = np.array(lengths)
    lengths = lengths <= threshold
    num = np.sum(lengths)
    print(num, 200)
    print(lengths)
    pass
    