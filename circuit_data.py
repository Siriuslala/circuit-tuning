"""
Dataset for circuit-tuning
Contains；
1. subject-verb disagreement
2. gender-bias
As for GMS8k, please refer to circuit-tuning/data/gms8k_dataset.py
"""

import torch as t
from torch.utils.data import Dataset, DataLoader
import jsonlines
from tqdm import tqdm

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(str(root_dir))


class SVDataset(Dataset):
    
    def __init__(self, tokenizer, split, data_num=None, use_contrastive=False):
        self.data = []
        self.tokenizer = tokenizer
        if split == "train":
            data_path = root_dir / "data/sv_dataset/train_24000.jsonl"
        elif split == "val" or split == "dev":
            data_path = root_dir / "data/sv_dataset/dev_3000.jsonl"
            if use_contrastive:
                data_path = root_dir / "data/sv_dataset/dev_3000_contrastive.jsonl"
        else:
            data_path = root_dir / "data/sv_dataset/test_3000.jsonl"
        
        cnt = 0
        with jsonlines.open(data_path, "r") as f:
            for line in f:
                self.data.append(line)
                cnt += 1
                if data_num is not None and cnt > data_num:
                    break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class SVCollateFn():
    
    def __init__(self, tokenizer, clean_corrupted_together=False, task="svd"):
        self.tokenizer = tokenizer
        self.clean_corrupted_together = clean_corrupted_together
        self.task = task
    
    def __call__(self, samples):
        # load data
        # {"clean_text", "corr_text", "clean_verb_pos", "corr_verb_pos", "clean_verb_ids", "corr_verb_ids", "clean_verbs", "corr_verbs"} 
        clean_text = [item["clean_text"] for item in samples]
        corr_text = [item["corr_text"] for item in samples]
        clean_verb_pos = [item["clean_verb_pos"] for item in samples]
        corr_verb_pos = [item["corr_verb_pos"] for item in samples]
        clean_verb_ids = [item["clean_verb_ids"] for item in samples]
        corr_verb_ids = [item["corr_verb_ids"] for item in samples]
        
        # tokenize
        if not self.clean_corrupted_together:
            clean_inputs = self.tokenizer.batch_encode_plus(clean_text, padding=True, return_tensors='pt')
            corr_inputs = self.tokenizer.batch_encode_plus(corr_text, padding=True, return_tensors='pt')
            if self.task == "svd":
                batch = {
                    "clean_inputs": corr_inputs,
                    "corr_inputs": clean_inputs,
                    "clean_verb_pos": corr_verb_pos,
                    "corr_verb_pos": clean_verb_pos,
                    "clean_verb_ids": corr_verb_ids,
                    "corr_verb_ids": clean_verb_ids,
                }  # sv disagreement
            else:
                batch = {
                    "clean_inputs": corr_inputs,
                    "corr_inputs": clean_inputs,
                    "clean_verb_pos": corr_verb_pos,
                    "corr_verb_pos": clean_verb_pos,
                    "clean_verb_ids": corr_verb_ids,
                    "corr_verb_ids": clean_verb_ids,
                }  # sv disagreement
        else:
            all_text = []
            for i in range(len(clean_text)):
                all_text.append(corr_text[i])
                all_text.append(clean_text[i])
            all_inputs = self.tokenizer.batch_encode_plus(all_text, padding=True, return_tensors='pt')
            batch = {
                "all_inputs": all_inputs,
                "clean_verb_pos": corr_verb_pos,
                "corr_verb_pos": clean_verb_pos,
                "clean_verb_ids": corr_verb_ids,
                "corr_verb_ids": clean_verb_ids,
            }  # sv disagreement

        return batch


class BiasDataset(Dataset):
    
    def __init__(self, tokenizer, data_num=None):
        self.data = []
        self.tokenizer = tokenizer
        data_path = root_dir / "data/bias/gender_bias/gender_bias_formatted.jsonl"
        
        cnt = 0
        with jsonlines.open(data_path, "r") as f:
            for line in f:
                self.data.append(line)
                cnt += 1
                if data_num is not None and cnt > data_num:
                    break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class BiasCollateFn():
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, samples):
        # load data
        # {"clean_text", "corr_text", "clean_verb_pos", "corr_verb_pos", "clean_verb_ids", "corr_verb_ids", "clean_verbs", "corr_verbs"} 
        prompts = [item["sentence"] for item in samples]
        pronoun_positions = [item["pronoun_pos"] for item in samples]
        pronoun_ids = [item["pronoun_id"] for item in samples]
        pronoun_anti_ids = [item["pronoun_anti_id"] for item in samples]
        
        # tokenize
        inputs = self.tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')
        
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pronoun_positions": pronoun_positions,
            "pronoun_ids": pronoun_ids,
            "pronoun_anti_ids": pronoun_anti_ids,
        }  # gender bias
        
        return batch


if __name__ == "__main__":
    from transformer_lens import HookedTransformer
    
    def test_sv_dataset():
        path = root_dir / "data/sv_pile_10k_formatted.jsonl"
        model = HookedTransformer.from_pretrained(
            'gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
        )
        dataset = SVDataset(path, model.tokenizer)
        # print(dataset[:-1])
        
        sv_collate_fn = SVCollateFn(model.tokenizer)
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=sv_collate_fn)
        for batch in tqdm(dataloader):
            print(batch)
            break
    
    def test_math_dataset():
        pass
