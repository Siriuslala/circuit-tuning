"""
Metrics for activation patching / attribution patching...
"""

import torch as t
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from circuit_data import SVDataset


def avg_logit_diff_sv(
    logits: Float[Tensor, 'batch seq d_vocab'],
    batch: dict,
    is_clean: bool = True,
    per_prompt: bool = False
):
    '''
    Calculate the average logit difference for the subject-verb agreement task.
    Return average logit difference between correct and incorrect answers.
    '''
    # Get logits for: verbs agreeing with subject & verbs not agreeing with subject
    # logits: [batch_size, seq_len, d_vocab]
    
    # clean inputs: "I likes and loves"  corr inputs: "I like and love"
    # clean_verbs: ["likes", "loves"]
    # corr_verbs: ["like", "love"]
    
    if is_clean:
        verb_pos = batch["clean_verb_pos"]
    else:
        verb_pos = batch["corr_verb_pos"]
        
    pre_word_pos = [[pos - 1 for pos in data] for data in verb_pos] 

    clean_verb_ids = batch["clean_verb_ids"]
    corr_verb_ids = batch["corr_verb_ids"]
    
    # print("logits:", logits.shape)
    # print("pre_word_pos:", pre_word_pos)
    # print("clean_verb_ids:", clean_verb_ids)
    # clean_v_logits = logits[:, pre_word_pos, clean_verb_ids]
    # corr_v_logits = logits[:, pre_word_pos, corr_verb_ids]
    
    logit_diff = []
    for i in range(logits.shape[0]):  # for each data sample
        clean_v_logits = logits[i, pre_word_pos[i], clean_verb_ids[i]]  # [num_verbs]
        corr_verb_logits = logits[i, pre_word_pos[i], corr_verb_ids[i]]
        logit_diff.append((clean_v_logits - corr_verb_logits).mean())  # average over verbs
    
    logit_diff = t.stack(logit_diff)  # [batch_size]
    
    return logit_diff if per_prompt else logit_diff.mean()

def avg_log_prob_sum_math(
    logits: Float[Tensor, 'batch seq d_vocab'],
    batch: dict,
    per_prompt: bool = False
):
    '''
    Calculate the average logit difference for the gender bias task.
    Return average logit difference between correct and incorrect answers.
    '''
    # Get logits for: verbs agreeing with subject & verbs not agreeing with subject
    # logits: [batch_size, seq_len, d_vocab]
    
    answer_positions = batch["answer_positions"]
    answer_ids = batch["answer_ids"]
    pre_answer_positions = [[pos -1 for pos in answer_pos] for answer_pos in answer_positions]
    
    probs = -t.log(t.nn.functional.softmax(logits, dim=-1))
    
    log_prob_sums = [] 
    for i in range(probs.shape[0]):  # for each data sample
        sum = probs[i, pre_answer_positions, answer_ids].sum()
        log_prob_sums.append(sum)
    
    log_prob_sums = t.stack(log_prob_sums)  # [batch_size]
    
    return log_prob_sums if per_prompt else log_prob_sums.mean()

def avg_logit_diff_bias(
    logits: Float[Tensor, 'batch seq d_vocab'],
    batch: dict,
    per_prompt: bool = False
):
    '''
    Calculate the average logit difference for the gender bias task.
    Return average logit difference between correct and incorrect answers.
    '''
    # Get logits for: verbs agreeing with subject & verbs not agreeing with subject
    # logits: [batch_size, seq_len, d_vocab]
    
    pronoun_positions = batch["pronoun_positions"]
    pronoun_ids = batch["pronoun_ids"]
    pronoun_anti_ids = batch["pronoun_anti_ids"]
        
    logit_diff = []
    for i in range(logits.shape[0]):  # for each data sample
        target_logits = logits[i, pronoun_positions[i] - 1]
        pronoun_logits = target_logits[pronoun_ids[i]]
        pronoun_anti_logits = target_logits[pronoun_anti_ids[i]]
        logit_diff.append(pronoun_logits - pronoun_anti_logits)
    
    logit_diff = t.stack(logit_diff)  # [batch_size]
    
    return logit_diff if per_prompt else logit_diff.mean()

def avg_neg_log_prob_diff_sv(
    logits: Float[Tensor, 'batch seq d_vocab'],
    batch: dict,
    is_clean: bool = True,
    per_prompt: bool = False
):
    '''
    Calculate the average log prob difference for the subject-verb agreement task.
    Return average log prob difference between correct and incorrect answers.
    '''
    # logits -> probs
    probs = t.nn.functional.softmax(logits, dim=-1)
    
    prob_diff = avg_logit_diff_sv(-t.log(probs), batch, is_clean, per_prompt)
    
    return prob_diff

def patching_metric_normalized(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    clean_logits: float,
    corrupted_logits: float,
    batch: dict,
    logit_diff_func: callable
 ):
    '''
    Metric for IE_AtP.
    Linear function of logit diff, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    clean_logit_diff = logit_diff_func(clean_logits, batch).item()
    corrupted_logit_diff = logit_diff_func(corrupted_logits, batch, is_clean=False).item()
    patched_logit_diff = logit_diff_func(logits, batch)
    
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


def patching_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    clean_logits: float,
    corrupted_logits: float,
    batch: dict,
    logit_diff_func: callable
 ):
    '''
    Metric for IE_AtP.
    Linear function of logit diff, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = logit_diff_func(logits, batch)
    
    return patched_logit_diff


def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -patching_metric(logits)
    
# # Get clean and corrupt logit differences
# with t.no_grad():
#     clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
#     corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)

# print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
# print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')