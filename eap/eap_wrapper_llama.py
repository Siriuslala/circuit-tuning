import gc
from functools import partial
from typing import Callable, Union, Literal

import einops
import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eap.eap_graph_old import EAPGraph
from eap.eap_graph_llama import EAPGraphLlama

import sys



upstream_activations_difference = None
# fwd_handles = {}
# bwd_handles = {}


"""
For EAP, we only need clean inputs.
"""

def EAP_clean_forward_hook_bias(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: EAPGraph,
    batch: None,
    ie_over_seq: bool = True,
    process_or_outcome: Literal["process", "outcome"] = "outcome",
    ablation_method: Literal["mean", "zero"] = None,
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    """
   
    if ie_over_seq:
        global upstream_activations_difference
    else:
        global upstream_activations_difference
        
    hook_slice = graph.get_hook_slice(hook.name)
    
    pronoun_pos = batch["pronoun_positions"]
    pre_word_pos = [pos - 1 for pos in pronoun_pos]
        
    if activations.ndim == 3:
        # [batch_size, seq_len, d_model] ~ hook_result
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(-2)  # [seq_len, d_model]  mean over items (other items as corrupted)
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2) - mean_act
        else:  
            # only look at the logits of the pre-verb token
            batch_size, seq_len, d_model = activations.shape
            act = torch.zeros((batch_size, d_model), device=activations.device)  # [batch_size, d_model]
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # act[i] mean over tokens: [d_model]
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(0)  # [d_model]  mean over samples -> add bacth dimension -> add hook dimension
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2) - mean_act
    
    elif activations.ndim == 4:
        # [batch_size, seq_len, n_heads, d_model] ~ hook_q
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).mean(dim=-1)  # [1, seq_len, n_heads]  mean over samples (other items as corrupted) -> add batch dimension: [1, seq_len, n_heads, d_model] -> mean over d_model: [1, seq_len, n_heads]
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1) - mean_act  # [bsz, len, n_heads]
        else:
            batch_size, seq_len, n_head, d_model = activations.shape
            act = torch.zeros((batch_size, n_head, d_model), device=activations.device)
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]]  # act[i]: [n_heads]
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice] = act
            else:
                mean_act = activations.mean(dim=1).mean(dim=0).unsqueeze(0).mean(dim=-1)  # mean over tokens and samples -> add batch dimension: [1, n_heads, d_model] -> mean over d_model: [1, n_heads]
                upstream_activations_difference[:, hook_slice] = act.mean(dim=-1) - mean_act

def EAP_clean_backward_hook_bias(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: EAPGraph,
    batch,
    config,
    ie_over_seq: bool = True,
    process_or_outcome: Literal["process", "outcome"] = "outcome",
    smooth=False,
    will_do_bwd=True,
    upstream_activations_difference = None
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    """
    
    if not will_do_bwd:
        return
    
    if not graph.eap_scores.is_cuda:  # after the first backward in EAP, there are hooks in "loss" though we've done "model.reset_hooks()" 
        return
    
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    
    pronoun_pos = batch["pronoun_positions"]
    pre_word_pos = [pos - 1 for pos in pronoun_pos]
    
    if grad.ndim == 3:
        if ie_over_seq:
            grad_expanded = grad.unsqueeze(-2)
        else:
            new_grad = torch.zeros((grad.shape[0], grad.shape[-1]), device=grad.device)
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_word_pos[i]]  # [d_model]
            grad_expanded = new_grad.unsqueeze(-2)  # Shape: [batch_size, 1, d_model]
    else:
        # [batch_size, seq_len, n_heads, d_model]
        if ie_over_seq:
            grad_expanded = grad.mean(dim=-1)  # [batch_size, seq_len, n_heads]
        else:
            new_grad = torch.zeros((grad.shape[0], grad.shape[-2], grad.shape[-1]), device=grad.device)
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_word_pos[i]]  # mean over d_model: [n_heads, d_model]
            grad_expanded = new_grad  # Shape: [batch_size, n_heads]

    # we compute the mean over the batch_size and seq_len dimensions
    if ie_over_seq:
        result = torch.matmul(
            upstream_activations_difference[:, :, earlier_upstream_nodes_slice].unsqueeze(-1),  # Shape: [batch_size, seq_len, n_upstream_nodes, 1]
            grad_expanded.mean(dim=-1).unsqueeze(-2).to(dtype=torch.bfloat16)
  # Shape: [batch_size, seq_len, 1, n_heads]
        ).mean(dim=0).mean(dim=0).abs() # we sum over the batch_size and seq_len dimensions  [bsz, len, up, n_heads] -> [up, n_heads]
    else:
        # print("grad_expanded:", grad_expanded.dtype)
        # print("upstream_activations_difference:", upstream_activations_difference.dtype)
        result = torch.matmul(
            upstream_activations_difference[:, earlier_upstream_nodes_slice].unsqueeze(-1),  # Shape: [batch_size, n_upstream_nodes, 1]
            grad_expanded.mean(dim=-1).unsqueeze(-2).to(dtype=torch.bfloat16)  # Shape: [batch_size, 1, n_heads]
        ).mean(dim=0).abs() # we sum over the batch_size dimension
        # print("result:", result)
        # print("grad_expanded:", grad_expanded)

    if smooth:
        IE_smooth = config.beta_1 * graph.last_eap_score[earlier_upstream_nodes_slice, hook_slice] + (1 - config.beta_1) * result
        U_smooth = config.beta_2 * graph.last_u[earlier_upstream_nodes_slice, hook_slice] + (1 - config.beta_2) * (result - IE_smooth).abs()
        result = IE_smooth * U_smooth
        
        graph.last_eap_score[earlier_upstream_nodes_slice, hook_slice] = result
        graph.last_u_score[earlier_upstream_nodes_slice, hook_slice] = U_smooth
        
    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result  # chain rule
    del result, grad_expanded

def EAP_clean_forward_hook_bias_multi_step(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: EAPGraph,
    batch: None,
    ie_over_seq: bool = True,
    process_or_outcome: Literal["process", "outcome"] = "outcome",
    ablation_method: Literal["mean", "zero"] = None,
    will_do_bwd: bool = False,
    upstream_activations_difference = None
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    """
        
    hook_slice = graph.get_hook_slice(hook.name)
    
    pronoun_pos = batch["pronoun_positions"]
    pre_word_pos = [pos - 1 for pos in pronoun_pos]
        
    if activations.ndim == 3:
        # [batch_size, seq_len, d_model] ~ hook_result
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(-2)  # [seq_len, d_model]  mean over items (other items as corrupted)
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2) - mean_act
        else:  
            # only look at the logits of the pre-verb token
            batch_size, seq_len, d_model = activations.shape
            act = torch.zeros((batch_size, d_model), device=activations.device)  # [batch_size, d_model]
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # act[i] mean over tokens: [d_model]
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(0)  # [d_model]  mean over samples -> add bacth dimension -> add hook dimension
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2) - mean_act
    
    elif activations.ndim == 4:
        # [batch_size, seq_len, n_heads, d_model] ~ hook_q
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).mean(dim=-1)  # [1, seq_len, n_heads]  mean over samples (other items as corrupted) -> add batch dimension: [1, seq_len, n_heads, d_model] -> mean over d_model: [1, seq_len, n_heads]
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1) - mean_act  # [bsz, len, n_heads]
        else:
            batch_size, seq_len, n_head, d_model = activations.shape
            act = torch.zeros((batch_size, n_head, d_model), device=activations.device)
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]]  # act[i]: [n_heads, d_model]
            
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice] = act
            else:
                if will_do_bwd:
                    mean_act = upstream_activations_difference[:, hook_slice].mean(dim=0).unsqueeze(0) # [1, n_heads]  mean over samples-> add batch dimension: [1, n_heads]
                    upstream_activations_difference[:, hook_slice] = act.mean(dim=-1) - mean_act  # [bsz, n_heads]
                else:
                    # not ie_over_seq, mean ablation
                    upstream_activations_difference[:, hook_slice] = (upstream_activations_difference[:, hook_slice] + activations.mean(dim=-1).mean(dim=1)) / 2

def EAP_clean_forward_hook_math(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: Union[EAPGraph, EAPGraphLlama],
    batch: None,
    ie_over_seq: bool = True,
    ablation_method: Literal["mean", "zero"] = None,
    process_or_outcome: Literal["process", "outcome"] = "process"
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    For math task, we have two settings:
    (1) Process. In this situation, we must use ie_over_len to pay attention to every position in the seq.
    (2) Outcome. In this situation, we can only focus on the final results tokens, or focus on all positions in the sequence. 
    
    ie_over_seq, process_or_outcome="process"
    not ie_over_seq, process_or_outcome="outcome"
    ie_over_seq, process_or_outcome="outcome"
    
    If ie_over_seq and outcome-based, the Lm for patching should focus on the key words (final result tokens).
    """
    
    if ie_over_seq:
        global upstream_activations_difference  # (bsz, len, n_upstream_nodes, d_model)
    else:
        global upstream_activations_difference  # (bsz, n_upstream_nodes, d_model)
        
    hook_slice = graph.get_hook_slice(hook.name)

    if process_or_outcome == "outcome":
        outcome_pos = batch["answer_positions"]
        pre_outcome_pos = [[pos - 1 for pos in positions] for positions in outcome_pos]
        
    if activations.ndim == 3:  # (bsz, len, d_model)
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(-2)  # [seq_len, d_model]  mean over samples (other items as corrupted) -> add batch dimension -> add hook dimension: [1, seq_len, 1, d_model]
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2) - mean_act
        else:  
            # only look at the logits of the key tokens
            batch_size, seq_len, d_model = activations.shape
            act = torch.zeros((batch_size, d_model), device=activations.device)  # [batch_size, d_model]
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_outcome_pos[i]].mean(dim=0)  # act[i]: [d_model] (mean over outcome tokens)
                
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=1).mean(dim=0).unsqueeze(0).unsqueeze(0)  # [d_model]  mean over tokens and samples (other tokens as corrupted) -> add bacth dimension -> add hook dimension
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2) - mean_act  # act is [bsz, 1, d_model]
    
    elif activations.ndim == 4:  
        # (bsz, len, n_heads, d_model) ~ hook_result, hook_mlp_out
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).mean(dim=-1)  # [1, seq_len, n_heads]  mean over samples (other items as corrupted) -> add batch dimension: [1, seq_len, n_heads, d_model] -> mean over d_model: [1, seq_len, n_heads]
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1) - mean_act  # [bsz, len, n_heads]
        else:
            batch_size, seq_len, n_head, d_model = activations.shape
            act = torch.zeros((batch_size, n_head, d_model), device=activations.device)
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_outcome_pos[i]].mean(dim=0)  # act[i]: [n_heads, d_model] (mean over outcome tokens)
                
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.mean(dim=-1)
            else:
                mean_act = activations.mean(dim=1).mean(dim=0).unsqueeze(0).mean(dim=-1)  # mean over tokens and samples (other tokens as corrupted) -> add batch dimension: [1, n_heads, d_model] -> mean over d_model: [1, n_heads]
                upstream_activations_difference[:, hook_slice] = act.mean(dim=-1) - mean_act
        # print("acivation:", activations.shape)
        # print("mean_act:", mean_act)
        # print("upstream_activations_difference_ie_over_seq[:, :, hook_slice]:", upstream_activations_difference_ie_over_seq[:, :, hook_slice])

def EAP_clean_backward_hook_math(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: Union[EAPGraph, EAPGraphLlama],
    batch,
    config,
    ie_over_seq: bool = True,
    process_or_outcome: Literal["process", "outcome"] = "process",
    smooth=False,
    will_do_bwd=True,
    upstream_activations_difference = None
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    For math task, we have two settings:
    (1) Process. In this situation, we must use ie_over_len to pay attention to every position in the seq.
    (2) Outcome. In this situation, we can only focus on the final results tokens, or focus on all positions in the sequence. 
    
    ie_over_seq, process_or_outcome="process"
    not ie_over_seq, process_or_outcome="outcome"
    ie_over_seq, process_or_outcome="outcome"
    
    If ie_over_seq and outcome-based, the Lm for patching should focus on the key words (final result tokens).
    """
    # global bwd_handles
    
    if not will_do_bwd:
        # gc.collect()
        # device = torch.cuda.current_device()
        # with torch.cuda.device(device):
        #     torch.cuda.empty_cache()
        return
    
    if not graph.eap_scores.is_cuda:  # after the first backward in EAP, there are hooks in "loss" though we've done "model.reset_hooks()" 
        return
    
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    
    if process_or_outcome == "outcome":
        outcome_pos = batch["answer_positions"]
        pre_outcome_pos = [[pos - 1 for pos in positions] for positions in outcome_pos]
    
    if grad.ndim == 3:  # [batch_size, seq_len, n_heads]
        if ie_over_seq:
            grad_expanded = grad  # add hook dimension
        else:
            new_grad = torch.zeros((grad.shape[0], grad.shape[-1]), device=grad.device)
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_outcome_pos[i]].mean(dim=0)  # mean over outcome tokens: [d_model]
            grad_expanded = new_grad.unsqueeze(-2)  # Shape: [batch_size, 1, d_model]
            
    else:
        # [batch_size, seq_len, n_heads, d_head] ~ hook_q
        # [batch_size, seq_len, n_mlp_head, d_mlp_head] ~ hook_pre_linear [4, 138, 128, 64]
        if ie_over_seq:
            grad_expanded = grad.mean(dim=-1)  # [batch_size, seq_len, n_heads]
        else:
            batch_size, seq_len, n_head, d_model = grad.shape
            grad_expanded = torch.zeros((batch_size, n_head), device=grad.device)  # Shape: [batch_size, n_heads]
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_outcome_pos[i]].mean(dim=0).mean(dim=-1)  # mean over outcome tokens: [n_heads]

    # we compute the mean over the batch_size and seq_len dimensions
    if ie_over_seq:
        result = torch.matmul(
            upstream_activations_difference[:, :, earlier_upstream_nodes_slice].unsqueeze(-1),  # Shape: [batch_size, seq_len, n_upstream_nodes, 1]
            grad_expanded.unsqueeze(-2).to(dtype=torch.bfloat16)  # Shape: [batch_size, seq_len, 1, n_heads]
        ).mean(dim=0).mean(dim=0).abs() # we mean over the batch_size and seq_len dimensions  [bsz, len, up, n_heads] -> [up, n_heads]
    else:
        result = torch.matmul(
            upstream_activations_difference[:, earlier_upstream_nodes_slice].unsqueeze(-1),  # Shape: [batch_size, n_upstream_nodes, 1]
            grad_expanded.unsqueeze(-2).to(dtype=torch.bfloat16) # Shape: [batch_size, 1, n_heads]
        ).mean(dim=0).abs() # we mean over the batch_size dimension  [bsz, up, n_heads] -> [up, n_heads]
    # print("grad_expanded:", grad_expanded)
    # print("result:", result)
    del grad, grad_expanded

    if smooth:
        IE_smooth = config.beta_1 * graph.last_eap_score[earlier_upstream_nodes_slice, hook_slice] + (1 - config.beta_1) * result
        U_smooth = config.beta_2 * graph.last_u[earlier_upstream_nodes_slice, hook_slice] + (1 - config.beta_2) * (result - IE_smooth).abs()
        result = IE_smooth * U_smooth
        
        graph.last_eap_score[earlier_upstream_nodes_slice, hook_slice] = result
        graph.last_u_score[earlier_upstream_nodes_slice, hook_slice] = U_smooth
    # print(hook.name, 
    #       "grad:", grad.shape,
    #       "grad_expanded:", grad_expanded.shape, 
    #       "result:", result.shape, 
    #       "early slice:", earlier_upstream_nodes_slice,
    #       "hook_slice:", hook_slice,
    #       "receiver:", graph.eap_scores[earlier_upstream_nodes_slice, hook_slice].shape) 
    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result  # chain rule
    
    # bwd_handles[hook.name].remove()

def EAP_clean_forward_hook_math_multi_step(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: Union[EAPGraph, EAPGraphLlama],
    batch: None,
    ie_over_seq: bool = True,
    ablation_method: Literal["mean", "zero"] = None,
    process_or_outcome: Literal["process", "outcome"] = "process",
    will_do_bwd: bool = False,
    upstream_activations_difference = None
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    For math task, we have two settings:
    (1) Process. In this situation, we must use ie_over_len to pay attention to every position in the seq.
    (2) Outcome. In this situation, we can only focus on the final results tokens, or focus on all positions in the sequence. 
    
    ie_over_seq, process_or_outcome="process"
    not ie_over_seq, process_or_outcome="outcome"
    ie_over_seq, process_or_outcome="outcome"
    
    If ie_over_seq and outcome-based, the Lm for patching should focus on the key words (final result tokens).
    """
    # global fwd_handels
    
    hook_slice = graph.get_hook_slice(hook.name)

    if process_or_outcome == "outcome":
        outcome_pos = batch["answer_positions"]
        pre_outcome_pos = [[pos - 1 for pos in positions] for positions in outcome_pos]
        
    if activations.ndim == 3:  # (bsz, len, d_model)
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice, :] = activations.unsqueeze(-2)
            else:
                if will_do_bwd:
                    mean_act = upstream_activations_difference[:, :, hook_slice].mean(dim=0).unsqueeze(0) # [1, len, n_heads]  mean over samples (other items as corrupted) -> add batch dimension: [1, 1, n_heads]
                    upstream_activations_difference[:, :, hook_slice] = activations - mean_act  # [bsz, len, n_heads]
                else:
                    # mean over d_model and tokens: [bsz, n_heads] -> add len dimension: [bsz, 1, n_heads]
                    # mean: [bsz+bsz, len, n_heads] / 2
                    upstream_activations_difference[:, :, hook_slice] = (upstream_activations_difference[:, :, hook_slice] + activations.mean(dim=1).unsqueeze(1)) / 2
        else:  
            # only look at the logits of the key tokens
            batch_size, seq_len, d_model = activations.shape
            act = torch.zeros((batch_size, d_model), device=activations.device)  # [batch_size, d_model]
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_outcome_pos[i]].mean(dim=0)  # act[i]: [d_model] (mean over outcome tokens)
                
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=1).mean(dim=0).unsqueeze(0).unsqueeze(0)  # [d_model]  mean over tokens and samples (other tokens as corrupted) -> add bacth dimension -> add hook dimension
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2) - mean_act  # act is [bsz, 1, d_model]
    
    elif activations.ndim == 4:  
        # (bsz, len, n_heads, d_model) ~ hook_result, hook_mlp_out
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1)
            else:
                if will_do_bwd:
                    mean_act = upstream_activations_difference[:, :, hook_slice].mean(dim=0).unsqueeze(0) # [1, len, n_heads]  mean over samples (other items as corrupted) -> add batch dimension: [1, 1, n_heads]
                    upstream_activations_difference[:, :, hook_slice] = activations.mean(dim=-1) - mean_act  # [bsz, len, n_heads]
                else:
                    # mean over d_model and tokens: [bsz, n_heads] -> add len dimension: [bsz, 1, n_heads]
                    # mean: [bsz+bsz, len, n_heads] / 2
                    upstream_activations_difference[:, :, hook_slice] = (upstream_activations_difference[:, :, hook_slice] + activations.mean(dim=-1).mean(dim=1).unsqueeze(1)) / 2
        else:
            batch_size, seq_len, n_head, d_model = activations.shape
            act = torch.zeros((batch_size, n_head, d_model), device=activations.device)
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_outcome_pos[i]].mean(dim=0)  # act[i]: [n_heads, d_model] (mean over outcome tokens)
                
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.mean(dim=-1)
            else:
                mean_act = activations.mean(dim=1).mean(dim=0).unsqueeze(0).mean(dim=-1)  # mean over tokens and samples (other tokens as corrupted) -> add batch dimension: [1, n_heads, d_model] -> mean over d_model: [1, n_heads]
                upstream_activations_difference[:, hook_slice] = act.mean(dim=-1) - mean_act
               
        # print("acivation:", activations.shape)
        # print("mean_act:", mean_act)
        # print("upstream_activations_difference_ie_over_seq[:, :, hook_slice]:", upstream_activations_difference_ie_over_seq[:, :, hook_slice])
    # fwd_handels[hook.name].remove()
# ============= EAP =============


def EAP_ablation(
    model: HookedTransformer,
    graph: EAPGraph,
    batch: dict,
    metric: Callable,
    config,
    upstream_activations_difference=None,
):
    """
    calculate IE (eap scores)
    Args:
    config:
        ablation_method: Literal["contrastive", "mean", "zero"] = None
        ie_over_seq
        process_or_outcome
        smooth: whether to smooth IEs
    """
    forward_hook_func_map = {
        "bias": EAP_clean_forward_hook_bias,
        "math": EAP_clean_forward_hook_math
    }
    backward_hook_func_map = { 
        "bias": EAP_clean_backward_hook_bias,
        "math": EAP_clean_backward_hook_math
    }
 
    if config.task in ["math", "bias"]:
        clean_input_ids = batch["input_ids"].to(config.device)
        clean_attention_mask = batch["attention_mask"].to(config.device)
    else:
        clean_input_ids = batch["clean_inputs"]["input_ids"].to(config.device)
        clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(config.device)
    
    # assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    batch_size, seq_len = clean_input_ids.shape[0], clean_input_ids.shape[1]

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))
    
    forward_hook = forward_hook_func_map[config.task]
    backward_hook = backward_hook_func_map[config.task]
    
    clean_upstream_hook_fn = partial(
        forward_hook,
        graph=graph,
        batch=batch,
        ie_over_seq=config.ie_over_seq,
        process_or_outcome=config.process_or_outcome,
        ablation_method=config.ablation_method
    )

    clean_downstream_hook_fn = partial(
        backward_hook,
        graph=graph,
        batch=batch,
        config=config,
        ie_over_seq=config.ie_over_seq,
        process_or_outcome=config.process_or_outcome,
        smooth=config.smooth,
    )

    # we perform a forward and backward pass on the clean input
    model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
    model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

    logits, loss = model(input=clean_input_ids, attention_mask=clean_attention_mask, return_type="both")

    # In the subject-verb agreement task, the patching metric Lm is different from the NLL loss for pretraining. So we need to zero_grad and re-backward on the NLL loss after patching. 
    if config.process_or_outcome == "outcome":
        value = metric(logits=logits, 
                   clean_logits=logits, 
                   corrupted_logits=None, 
                   batch=batch)
        value.backward(retain_graph=True)  # save computational graph for the next backward
    # While in the math task, Lm is equal to the NLL loss, so we don't need to zero_grad and re-backward.
    else:
        loss.requires_grad_(True)
        loss.backward()
    
    if config.task == "bias":
        loss += abs(value) * config.regularization_beta

    if config.ie_over_seq:
        del upstream_activations_difference
    else:
        del upstream_activations_difference
    
    gc.collect()
    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= batch_size
    graph.eap_scores = graph.eap_scores.cpu()
    # print("EAP scores:", graph.eap_scores)

    return graph, logits, loss

def EAP_ablation_multi_step(
    model: HookedTransformer,
    graph: EAPGraph,
    batch: dict,
    metric: Callable,
    config,
    do_bwd=False,
    upstream_activations_difference=None,
):
    """
    calculate IE (eap scores)
    Args:
    config:
        ablation_method: Literal["contrastive", "mean", "zero"] = None
        ie_over_seq
        process_or_outcome
        smooth: whether to smooth IEs
    """
    forward_hook_func_map = {
        "bias": EAP_clean_forward_hook_bias_multi_step,
        "math": EAP_clean_forward_hook_math_multi_step,
        "logic": EAP_clean_forward_hook_math_multi_step,
        "reading": EAP_clean_forward_hook_math_multi_step
    }
    backward_hook_func_map = { 
        "bias": EAP_clean_backward_hook_bias,
        "math": EAP_clean_backward_hook_math,
        "logic": EAP_clean_backward_hook_math,
        "reading": EAP_clean_backward_hook_math
    }
    
    device = torch.cuda.current_device()
    if config.task in ["math", "bias", "logic", "reading"]:
        clean_input_ids = batch["input_ids"].to(device)
        clean_attention_mask = batch["attention_mask"].to(device)
    else:
        clean_input_ids = batch["clean_inputs"]["input_ids"].to(config.device)
        clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(config.device)
    
    # assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    batch_size, seq_len = clean_input_ids.shape[0], clean_input_ids.shape[1]
    if do_bwd and config.ablation_method == "mean" and config.ie_over_seq:
        upstream_activations_difference = upstream_activations_difference.expand(-1, seq_len, -1).clone()

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))
    
    forward_hook = forward_hook_func_map[config.task]
    backward_hook = backward_hook_func_map[config.task]
    
    clean_upstream_hook_fn = partial(
        forward_hook,
        graph=graph,
        batch=batch,
        ie_over_seq=config.ie_over_seq,
        process_or_outcome=config.process_or_outcome,
        ablation_method=config.ablation_method,
        will_do_bwd=do_bwd,
        upstream_activations_difference=upstream_activations_difference
    )

    clean_downstream_hook_fn = partial(
        backward_hook,
        graph=graph,
        batch=batch,
        config=config,
        ie_over_seq=config.ie_over_seq,
        process_or_outcome=config.process_or_outcome,
        smooth=config.smooth,
        will_do_bwd=do_bwd,
        upstream_activations_difference=upstream_activations_difference
    )
    

    # we perform a forward pass on the clean input
    # global fwd_handels
    # fwd_handels = model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
    # print("fwd_handels:", fwd_handels)
    model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
    
    logits, loss = model(input=clean_input_ids, attention_mask=clean_attention_mask, return_type="both")
    model.reset_hooks()
    
    if not do_bwd:
        # gc.collect()
        # with torch.cuda.device(device):
        #     torch.cuda.empty_cache()
        return logits, loss, upstream_activations_difference
    
    # we perform a backward pass on the clean input
    # global bwd_handles
    # bwd_handles = model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")
    model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

    # check if graph.eap_scores is all zero
    past_scores = None
    if graph.eap_scores.sum() != 0:
        past_scores = graph.eap_scores.clone()
        graph.reset_scores()
    
    # In the subject-verb agreement task, the patching metric Lm is different from the NLL loss for pretraining. So we need to zero_grad and re-backward on the NLL loss after patching. 
    if config.process_or_outcome == "outcome":
        value = metric(logits=logits, 
                   clean_logits=logits, 
                   corrupted_logits=None, 
                   batch=batch)
        value.backward()  # save computational graph for the next backward
    # While in the math task, Lm is equal to the NLL loss, so we don't need to zero_grad and re-backward.
    else:
        loss.requires_grad_(True)
        loss.backward()
    
    if config.task == "bias":
        loss += abs(value) * config.regularization_beta
    
    del upstream_activations_difference
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= batch_size
    graph.eap_scores = graph.eap_scores.cpu()
    if past_scores is not None:
        graph.eap_scores += past_scores
    # print("EAP scores:", graph.eap_scores)

    return graph, logits, loss


if __name__ == "__main__":
    
    from circuit_data import SVDataset, SVCollateFn
    from patching_metrics import patching_metric, avg_logit_diff_sv
    from torch.utils.data import DataLoader
    import dataclasses
    import numpy as np
    
    @dataclasses.dataclass
    class Config:
        device: str = "cuda"
        beta_1: float = 0.9
        beta_2: float = 0.9
    
    config = Config()
    
    model = HookedTransformer.from_pretrained(
        'gpt2-small',
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device="cuda"
    )
    
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    
    
    def compare_sv_svd():
        """
        Compare the sv circuit adn svd circuit
        """
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
        graph.reset_scores()
        patching_metric_fn = partial(patching_metric, logit_diff_func=avg_logit_diff_sv)
    
        train_dataset = SVDataset("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/train_24000.jsonl", model.tokenizer)
        sv_collate_fn = SVCollateFn(model.tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=sv_collate_fn)
        
        for batch in dataloader:
            new_batch = {
                "clean_inputs": batch["corr_inputs"],
                "corr_inputs": batch["clean_inputs"],
                "clean_verb_pos": batch["corr_verb_pos"],
                "corr_verb_pos": batch["clean_verb_pos"],
                "clean_verb_ids": batch["corr_verb_ids"],
                "corr_verb_ids": batch["clean_verb_ids"],
            }  # sv agreement
            graph, _, _ = EAP_ablation(
                model,
                graph=graph,
                batch=new_batch,
                metric=patching_metric_fn,
                config=config,
                smooth=None,
                )
        
            top_n = 35
            top_edges = graph.top_edges(n=top_n, threshold=0, abs_scores=True)
            
            
            # get svd scores
            import jsonlines
            log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-graph/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_50-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
            steps = []
            top_edges_svd = None
            with jsonlines.open(log_path, "r") as f:
                for line in f:
                    if "edges" in line:
                        step = line["step"]
                        while step in steps:
                            step += 1500
                        steps.append(step)
                        if step == 4000:  # 4000 3800 3400
                            top_edges_svd = line["edges"]
                            break
            
            # log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_50-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
            # data = []
            # with jsonlines.open(log_path, "r") as f:
            #     for line in f:
            #         data.append(line)
            # top_edges_svd = data[-1]["edge_info"]
            
            top_edges_svd = sorted(top_edges_svd, key=lambda x: x[-1], reverse=True)[:top_n]
            
            # get log scores
            for edge in top_edges_svd:
                edge[-1] = np.log(edge[-1] + 1)
        

            # get log scores
            for i in range(len(top_edges)):
                top_edges[i] = [top_edges[i][0], top_edges[i][1], np.log(top_edges[i][2] + 1)]

            
            sv_nodes = set()
            svd_nodes = set()
            sv_edges = set()
            svd_edges = set()
            for edge in top_edges:
                up, down, score = edge
                down = down.replace(".q", "").replace(".k", "").replace(".v", "")
                sv_nodes.add(up)
                sv_nodes.add(down)
                sv_edges.add((up, down))
            for edge in top_edges_svd:
                up, down, score = edge
                down = down.replace(".q", "").replace(".k", "").replace(".v", "")
                svd_nodes.add(up)
                svd_nodes.add(down)
                svd_edges.add((up, down))
            for i, edge in enumerate(top_edges_svd):
                up, down, score = edge
                if "head.0.5" in up:
                    break
            top_edges_svd.pop(i)
            
            # intersection
            common_nodes = list(sv_nodes.intersection(svd_nodes))
            common_edges = list(sv_edges.intersection(svd_edges))
            print("common edges:", common_edges)

            key_nodes = common_nodes
            key_edges = common_edges
            _, _ = graph.show(edges=top_edges, key_nodes=key_nodes, key_edges=key_edges, fname=f"eap_graph_sv_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_sv_svd")
            _, _ = graph.show(edges=top_edges_svd, key_nodes=key_nodes, key_edges=key_edges, fname=f"svd_graph_svd_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_sv_svd")
            break
    
    
    def get_sv_graph():
        
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
        graph.reset_scores()
        patching_metric_fn = partial(patching_metric, logit_diff_func=avg_logit_diff_sv)
        
        train_dataset = SVDataset("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/train_24000.jsonl", model.tokenizer)
        sv_collate_fn = SVCollateFn(model.tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=sv_collate_fn)
        
        top_edges = None
        for batch in dataloader:
            new_batch = {
                "clean_inputs": batch["corr_inputs"],
                "corr_inputs": batch["clean_inputs"],
                "clean_verb_pos": batch["corr_verb_pos"],
                "corr_verb_pos": batch["clean_verb_pos"],
                "clean_verb_ids": batch["corr_verb_ids"],
                "corr_verb_ids": batch["clean_verb_ids"],
            }  # sv agreement
            graph, _, _ = EAP_ablation(
                model,
                graph=graph,
                batch=new_batch,
                metric=patching_metric_fn,
                config=config,
                smooth=None,
                )
        
            top_n = 200
            top_edges = graph.top_edges(n=top_n, threshold=0, abs_scores=True)
            break
        
        top_edges = sorted(top_edges, key=lambda x: x[-1], reverse=True)[:]
        
        _, edges = graph.show(edges=top_edges, fname=f"eap_graph_sv_topn_{top_n}_sel_35.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims")

        edge_info_path = os.path.join("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims", "sv_topn_50_sel_35_edge_info.jsonl")
        import jsonlines
        with jsonlines.open(edge_info_path, "w") as f:
            f.write({"edges": edges, "name": "eap_graph_sv_topn_50"})
    
    get_sv_graph()
        