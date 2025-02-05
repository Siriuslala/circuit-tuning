"""
Circuit-tuning
"""

import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils
from transformer_lens.train import train
from circuit_data import SVDataset, SVCollateFn, MathDataset, MathCollateFn
from circuit_evaluation import evaluate_sv, evaluate_math

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

import argparse
from tqdm import tqdm
import wandb
from typing import Optional
import os
import sys
from functools import partial

import logging
import jsonlines
    

def weighted_loss_func(logits,
                       tokens, 
                       attention_mask, 
                       verb_pos, 
                       loss_weight_p, 
                       verb_alone=False):
    """
    p(w1)p(w2)...p(verb)...p(wn) ->
    p(w1)p(w2)...p(verb)^{1/p}...p(wn), p \in (0, 1]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]

    # Ignore token positions which are masked out or where the next token is masked out
    # (generally padding tokens)
    next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
    predicted_log_probs *= next_token_mask  #  [batch_size, seq_len - 1]
    n_tokens = next_token_mask.sum().item()
    
    pre_verb_pos = [[pos - 1 for pos in data] for data in verb_pos] 
    if verb_alone:
        verb_probs = []
        for i in range(len(pre_verb_pos)):
            verb_probs.append(predicted_log_probs[i, pre_verb_pos[i]])
        loss = torch.cat([verb_prob.unsqueeze(0)  for verb_prob in verb_probs], dim=1).sum()
        total_verb_num = sum([len(data) for data in pre_verb_pos])
        loss /= total_verb_num
    else:
        for i in range(len(pre_verb_pos)):
            predicted_log_probs[i, pre_verb_pos[i]] *= 1 / loss_weight_p
        loss = -predicted_log_probs.sum() / n_tokens

    return loss

def train(
    config,
    model: HookedTransformer,
    dataset: Optional[SVDataset],
) -> HookedTransformer:
    
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.jsonl")
    
    torch.manual_seed(config.seed)
    
    if config.wandb:
        wandb.login(key=config.wandb_api_key)
        print(config.wandb_project_name)
        print(os.path.basename(save_dir))
        wandb.init(project=config.wandb_project_name, 
                   name=os.path.basename(save_dir), 
                   settings=wandb.Settings(start_method="fork"))

    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(config.weight_decay if config.weight_decay is not None else 0.0),
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )
    
    train_dataset, dev_dataset = dataset
    train_steps_per_epoch = len(train_dataset) // config.batch_size
    save_steps = config.save_every if config.save_every is not None else train_steps_per_epoch - 1

    if config.task == "sv":
        # subject-verb agreement
        sv_collate_fn = SVCollateFn(model.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=sv_collate_fn)
        eval_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=sv_collate_fn)
        eval_func = evaluate_sv
    elif config.task == "math":
        # GSM8k
        math_collate_fn = MathCollateFn(model.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=math_collate_fn)
        eval_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=math_collate_fn)
        eval_func = evaluate_math
        pass
    
    model.train()
    
    for epoch in range(1, config.num_epochs + 1):
        samples = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            
            # forward
            clean_input_ids = batch["clean_inputs"]["input_ids"].to(config.device)
            clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(config.device)
            logits, loss = model(input=clean_input_ids, 
                                 attention_mask=clean_attention_mask, 
                                 return_type="both")

                                   
            # backward and mask grads
            if config.objective == "standard" or config.loss_weight_p == 1:
                loss.backward()
            else:
                if config.objective == "weighted":
                    loss_weight_p = config.loss_weight_p
                    weighted_loss = weighted_loss_func(logits=logits,
                                              tokens=batch["clean_inputs"]["input_ids"].to(config.device), 
                                              attention_mask=batch["clean_inputs"]["attention_mask"].to(config.device), 
                                              verb_pos=batch["clean_verb_pos"], 
                                              loss_weight_p=loss_weight_p)
                else:
                    weighted_loss = weighted_loss_func(logits=logits, 
                                              tokens=batch["clean_inputs"]["input_ids"].to(config.device), 
                                              attention_mask=batch["clean_inputs"]["attention_mask"].to(config.device), 
                                              verb_pos=batch["clean_verb_pos"],
                                              loss_weight_p=None,
                                              verb_alone=True)
                weighted_loss.backward()
 
            
            # optimize
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if config.accumulation_steps > 1:
                if (step + 1) % config.accumulation_steps == 0:
                    optimizer.step()
                    if config.warmup_steps > 0:
                        assert scheduler is not None
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                if config.warmup_steps > 0:
                    assert scheduler is not None
                    scheduler.step()
                optimizer.zero_grad()
            
            # print and log
            samples += config.batch_size
            
            if config.print_every is not None and step > 0 and step % config.print_every == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item()}")
                sys.stdout.flush()

            lr = scheduler.get_last_lr()[0]
            
            # evaluation
            eval_info = {}
            if config.eval_steps is not None and step > 0 and step % config.eval_steps == 0:
                eval_result = eval_func(model, eval_dataloader, config)
                eval_info = {"eval_result": eval_result}
            
            basic_info = {"loss": loss.item(), "learning rate": lr}
            log_info = {**basic_info, **eval_info}
            with jsonlines.open(log_path, mode='a') as f:
                f.write(log_info)
            if config.wandb:
                wandb.log(log_info)
            
            # save
            if step > 0 and step % save_steps == 0:
                save_path = os.path.join(save_dir, f"model-steps_{step}_epoch_{epoch}.pt")
                torch.save(model.state_dict(), save_path)

            if config.max_steps is not None and step >= config.max_steps:
                break

    return model


if __name__ == '__main__':
    # get training args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-small')
    parser.add_argument('--task', type=str, default='sv')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--device', type=str,  default="cpu")
    parser.add_argument('--gpu_ids', type=int, nargs='+', help='List of GPU IDs', default="0")
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--optimizer_name', type=str, default='SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--eval_steps', type=int, default=100)
    
    parser.add_argument('--objective', type=str, choices=["standard", "weighted", "alone"], default="standard")
    parser.add_argument('--loss_weight_p', type=float, default=1.0)
    
    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_api_key', type=str, default=None)
    parser.add_argument('--wandb_project_name', type=str, default=None)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)  # /checkpoints/xxx
    
    args = parser.parse_args()
    print(args)
    
    # get model
    print('Loading model...')
    
    if args.device is None:
        device = utils.get_device()
    else:
        if args.device == "cpu":
            device = torch.device("cpu")
        else:
            gpu_ids = args.gpu_ids
            print(f"Using GPU {gpu_ids}")
            if len(gpu_ids) == 1:
                device = torch.device(f'cuda:{gpu_ids[0]}')
            else:
                device = "cuda"
            args.device = device
            
    model = HookedTransformer.from_pretrained(
        model_name=args.model,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=args.device
    )
    print("model.cfg.device:", model.cfg.device)
    
    
    # get data
    data_dir = args.data_dir
    data_files = os.listdir(data_dir)
    for file in data_files:
        if "train" in file:
            train_data_path = os.path.join(data_dir, file)
        elif "dev" in file:
            dev_data_path = os.path.join(data_dir, file)
    if args.task == "sv":
        train_dataset = SVDataset(train_data_path, model.tokenizer)
        dev_dataset = SVDataset(dev_data_path, model.tokenizer)
    elif args.task == "math":
        train_dataset = MathDataset(train_data_path, model.tokenizer)
        dev_dataset = MathDataset(dev_data_path, model.tokenizer)
    else:
        raise ValueError(f"Task {args.task} not supported")
    
    # train
    _ = train(args, model, (train_dataset, dev_dataset))
    print('Training done!')
    
    
    
    