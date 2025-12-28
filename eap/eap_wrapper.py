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



upstream_activations_difference_ie_over_seq = None
upstream_activations_difference = None


"""
For EAP_standard, we need contrastive inputs.
For example: "The teacher is" vs "The teacher are"
So we need a clean forward pass and a corrupted forward pass.

For EAP_ablation, we only need clean inputs.
"""
def EAP_standard_corrupted_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: Union[EAPGraph, EAPGraphLlama],
    batch: None,
):
    global upstream_activations_difference
    
    hook_slice = graph.get_hook_slice(hook.name)
    
    verb_pos = batch["corr_verb_pos"]
    pre_word_pos = [[pos - 1 for pos in data] for data in verb_pos]
    # act: [batch_size, d_model]
    
    if activations.ndim == 3:
        # We are in the case of a residual layer or MLP
        # Activations have shape [batch_size, seq_len, d_model]
        # We need to add an extra dimension to make it [batch_size, seq_len, 1, d_model]
        # The hook slice is a slice of length 1
        act = torch.zeros((activations.shape[0], activations.shape[-1]), device=activations.device)
        for i in range(activations.shape[0]):  # for each data sample
            act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # [d_model]
        upstream_activations_difference[:, hook_slice, :] = -act.unsqueeze(-2)
    
    elif activations.ndim == 4:
        # We are in the case of an attention layer
        # Activations have shape [batch_size, seq_len, n_heads, d_model]
        act = torch.zeros((activations.shape[0], activations.shape[-2], activations.shape[-1]), device=activations.device)
        for i in range(activations.shape[0]):  # for each data sample
            act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # [n_heads, d_model]
        upstream_activations_difference[:, hook_slice, :] = -act

def EAP_standard_clean_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: EAPGraph,
    batch: None,
):
    global upstream_activations_difference
    
    hook_slice = graph.get_hook_slice(hook.name)

    verb_pos = batch["clean_verb_pos"]
    pre_word_pos = [[pos - 1 for pos in data] for data in verb_pos]
        
    if activations.ndim == 3:
        act = torch.zeros((activations.shape[0], activations.shape[-1]), device=activations.device)
        for i in range(activations.shape[0]):  # for each data sample
            act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # [d_model]
        upstream_activations_difference[:, hook_slice, :] += act.unsqueeze(-2)
    
    elif activations.ndim == 4:
        act = torch.zeros((activations.shape[0], activations.shape[-2], activations.shape[-1]), device=activations.device)
        for i in range(activations.shape[0]):  # for each data sample
            act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # [n_heads, d_model]
        upstream_activations_difference[:, hook_slice, :] += act


"""
For EAP, we only need clean inputs.
"""
def EAP_clean_forward_hook_sv(
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
        global upstream_activations_difference_ie_over_seq
    else:
        global upstream_activations_difference
        
    hook_slice = graph.get_hook_slice(hook.name)
    
    verb_pos = batch["clean_verb_pos"]
    pre_word_pos = [[pos - 1 for pos in data] for data in verb_pos]
        
    if activations.ndim == 3:
        # [batch_size, seq_len, d_model] ~ hook_result
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(-2)  # [seq_len, d_model]  mean over items (other items as corrupted)
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations.unsqueeze(-2) - mean_act
        else:  
            # only look at the logits of the pre-verb token
            batch_size, seq_len, d_model = activations.shape
            act = torch.zeros((batch_size, d_model), device=activations.device)  # [batch_size, d_model]
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # act[i]: [d_model] (mean over verbs)
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=1).mean(dim=0)  # [d_model]  mean over tokens and items (other tokens as corrupted)
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2) - mean_act.unsqueeze(0).unsqueeze(0)  # act is [bsz, 1, d_model]
    
    elif activations.ndim == 4:
        # [batch_size, seq_len, n_heads, d_model] ~ hook_q
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0)
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations - mean_act
        else:
            batch_size, seq_len, n_head, d_model = activations.shape
            act = torch.zeros((batch_size, n_head, d_model), device=activations.device)
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]].mean(dim=0)  # act[i]: [n_heads, d_model] (mean over verbs)
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act
            else:
                mean_act = activations.mean(dim=1).mean(dim=0)  # [n_heads, d_model]
                upstream_activations_difference[:, hook_slice, :] = act - mean_act.unsqueeze(0)

def EAP_clean_backward_hook_sv(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: EAPGraph,
    batch,
    config,
    ie_over_seq: bool = True,
    process_or_outcome: Literal["process", "outcome"] = "outcome",
    smooth=False,
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    """
    
    if ie_over_seq:
        global upstream_activations_difference_ie_over_seq
    else:
        global upstream_activations_difference
    
    if not graph.eap_scores.is_cuda:  # after the first backward in EAP, there are hooks in "loss" though we've done "model.reset_hooks()" 
        return
    
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    
    verb_pos = batch["clean_verb_pos"]
    pre_word_pos = [[pos - 1 for pos in data] for data in verb_pos]
    
    if grad.ndim == 3:
        if ie_over_seq:
            grad_expanded = grad.unsqueeze(-2)
        else:
            new_grad = torch.zeros((grad.shape[0], grad.shape[-1]), device=grad.device)
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_word_pos[i]].mean(dim=0)  # [d_model]
            grad_expanded = new_grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
    else:
        if ie_over_seq:
            grad_expanded = grad
        else:
            new_grad = torch.zeros((grad.shape[0], grad.shape[-2], grad.shape[-1]), device=grad.device)
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_word_pos[i]].mean(dim=0)  # [d_model]
            grad_expanded = new_grad  # Shape: [batch_size, seq_len, n_heads, d_model]

    # we compute the mean over the batch_size and seq_len dimensions
    if ie_over_seq:
        result = torch.matmul(
            upstream_activations_difference_ie_over_seq[:, :, earlier_upstream_nodes_slice],  # Shape: [batch_size, seq_len, n_upstream_nodes, d_model]
            grad_expanded.transpose(-1, -2)  # Shape: [batch_size, seq_len, d_model, n_heads]
        ).mean(dim=0).mean(dim=0).abs() # we sum over the batch_size and seq_len dimensions
    else:
        result = torch.matmul(
            upstream_activations_difference[:, earlier_upstream_nodes_slice],
            grad_expanded.transpose(-1, -2)
        ).mean(dim=0).abs() # we sum over the batch_size dimension

    if smooth:
        IE_smooth = config.beta_1 * graph.last_eap_score[earlier_upstream_nodes_slice, hook_slice] + (1 - config.beta_1) * result
        U_smooth = config.beta_2 * graph.last_u[earlier_upstream_nodes_slice, hook_slice] + (1 - config.beta_2) * (result - IE_smooth).abs()
        result = IE_smooth * U_smooth
        
        graph.last_eap_score[earlier_upstream_nodes_slice, hook_slice] = result
        graph.last_u_score[earlier_upstream_nodes_slice, hook_slice] = U_smooth
        
    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result  # chain rule
    del result, grad_expanded


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
        global upstream_activations_difference_ie_over_seq
    else:
        global upstream_activations_difference
        
    hook_slice = graph.get_hook_slice(hook.name)
    
    pronoun_pos = batch["pronoun_positions"]
    pre_word_pos = [pos - 1 for pos in pronoun_pos]
        
    if activations.ndim == 3:
        # [batch_size, seq_len, d_model] ~ hook_result
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0).unsqueeze(-2)  # [seq_len, d_model]  mean over items (other items as corrupted)
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations.unsqueeze(-2) - mean_act
        else:  
            # only look at the logits of the pre-verb token
            batch_size, seq_len, d_model = activations.shape
            act = torch.zeros((batch_size, d_model), device=activations.device)  # [batch_size, d_model]
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]]  # act[i]: [d_model]
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2)
            else:
                mean_act = activations.mean(dim=1).mean(dim=0)  # [d_model]  mean over tokens and items (other tokens as corrupted)
                upstream_activations_difference[:, hook_slice, :] = act.unsqueeze(-2) - mean_act.unsqueeze(0).unsqueeze(0)  # act is [bsz, 1, d_model]
    
    elif activations.ndim == 4:
        # [batch_size, seq_len, n_heads, d_model] ~ hook_q
        if ie_over_seq:
            if ablation_method == "zero":
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations
            else:
                mean_act = activations.mean(dim=0).unsqueeze(0)
                upstream_activations_difference_ie_over_seq[:, :, hook_slice, :] = activations - mean_act
        else:
            batch_size, seq_len, n_head, d_model = activations.shape
            act = torch.zeros((batch_size, n_head, d_model), device=activations.device)
            for i in range(batch_size):  # for each data sample
                act[i] = activations[i, pre_word_pos[i]]  # act[i]: [n_heads, d_model]
            if ablation_method == "zero":
                upstream_activations_difference[:, hook_slice, :] = act
            else:
                mean_act = activations.mean(dim=1).mean(dim=0)  # [n_heads, d_model]
                upstream_activations_difference[:, hook_slice, :] = act - mean_act.unsqueeze(0)

def EAP_clean_backward_hook_bias(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    graph: EAPGraph,
    batch,
    config,
    ie_over_seq: bool = True,
    process_or_outcome: Literal["process", "outcome"] = "outcome",
    smooth=False,
):
    """
    ie over len: for calculating the activations and gradients, whether to calculate them over the sequence length or simply on the key word(token).
    """
    
    if ie_over_seq:
        global upstream_activations_difference_ie_over_seq
    else:
        global upstream_activations_difference
    
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
        if ie_over_seq:
            grad_expanded = grad
        else:
            new_grad = torch.zeros((grad.shape[0], grad.shape[-2], grad.shape[-1]), device=grad.device)
            for i in range(grad.shape[0]):  # for each data sample
                new_grad[i] = grad[i, pre_word_pos[i]]  # [n_heads, d_model]
            grad_expanded = new_grad  # Shape: [batch_size, n_heads, d_model]

    # we compute the mean over the batch_size and seq_len dimensions
    if ie_over_seq:
        result = torch.matmul(
            upstream_activations_difference_ie_over_seq[:, :, earlier_upstream_nodes_slice],  # Shape: [batch_size, seq_len, n_upstream_nodes, d_model]
            grad_expanded.transpose(-1, -2)  # Shape: [batch_size, seq_len, d_model, n_heads]
        ).mean(dim=0).mean(dim=0).abs() # we sum over the batch_size and seq_len dimensions
    else:
        result = torch.matmul(
            upstream_activations_difference[:, earlier_upstream_nodes_slice],  # Shape: [batch_size, n_upstream_nodes, d_model]
            grad_expanded.transpose(-1, -2)
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


# ============= EAP =============
forward_hook_func_map = {
    "sv": EAP_clean_forward_hook_sv,
    "bias": EAP_clean_forward_hook_bias,
}
backward_hook_func_map = { 
    "sv": EAP_clean_backward_hook_sv,
    "bias": EAP_clean_backward_hook_bias
}


def EAP_standard(
    model: HookedTransformer,
    graph: EAPGraph,
    batch: dict,
    metric: Callable,
    config,
    ie_over_seq: bool = False,
    smooth=False,
):
    # ablation_method: Literal["contrastive", "mean", "zero"] = None,
    """
    calculate IE (eap scores)
    Args:
    smooth: whether to smooth IEs
    """
    global upstream_activations_difference
    
    clean_input_ids = batch["clean_inputs"]["input_ids"].to(config.device)
    clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(config.device)
    corr_input_ids = batch["corr_inputs"]["input_ids"].to(config.device)
    corr_attention_mask = batch["corr_inputs"]["attention_mask"].to(config.device)
    
    # assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    batch_size, seq_len = clean_input_ids.shape[0], clean_input_ids.shape[1]

    upstream_activations_difference = torch.zeros(
        (batch_size, graph.n_upstream_nodes, model.cfg.d_model),
        device=config.device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    corruped_upstream_hook_fn = partial(
        EAP_standard_corrupted_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph,
        batch=batch
    )

    clean_upstream_hook_fn = partial(
        EAP_standard_clean_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph,
        batch=batch
    )

    clean_downstream_hook_fn = partial(
        EAP_clean_backward_hook_sv,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph,
        batch=batch,
        config=config,
        ie_over_seq=False,
        smooth=smooth,
    )

    # we first perform a forward pass on the corrupted input 
    model.add_hook(upstream_hook_filter, corruped_upstream_hook_fn, "fwd")

    # we don't need gradients for this forward pass
    # we'll take the gradients when we perform the forward pass on the clean input
    with torch.no_grad(): 
        corr_logits = model(input=corr_input_ids, attention_mask=corr_attention_mask, return_type="logits")        

    # now we perform a forward and backward pass on the clean input
    model.reset_hooks()
    model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
    model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

    logits, loss = model(input=clean_input_ids, attention_mask=clean_attention_mask, return_type="both")
    value = metric(logits=logits, 
                   clean_logits=logits, 
                   corrupted_logits=corr_logits, 
                   batch=batch)
    value.backward(retain_graph=True)  # to avoid "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad()"
    
    model.zero_grad()

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= batch_size
    graph.eap_scores = graph.eap_scores.cpu()
    # print("EAP scores:", graph.eap_scores)

    return graph, logits, loss


def EAP_ablation(
    model: HookedTransformer,
    graph: EAPGraph,
    batch: dict,
    metric: Callable,
    config,
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
    
    device = model.cfg.device
    if config.task in ["math", "bias"]:
        clean_input_ids = batch["input_ids"].to(device)
        clean_attention_mask = batch["attention_mask"].to(device)
    else:
        clean_input_ids = batch["clean_inputs"]["input_ids"].to(device)
        clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(device)
    
    # assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    batch_size, seq_len = clean_input_ids.shape[0], clean_input_ids.shape[1]
    
    if config.ie_over_seq:
        global upstream_activations_difference_ie_over_seq
        upstream_activations_difference_ie_over_seq  = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )
    else:
        global upstream_activations_difference
        upstream_activations_difference = torch.zeros(
        (batch_size, graph.n_upstream_nodes, model.cfg.d_model),
        device=device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

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
        model.zero_grad()
    # While in the math task, Lm is equal to the NLL loss, so we don't need to zero_grad and re-backward.
    else:
        loss.requires_grad_(True)
        loss.backward()
    
    if config.task == "bias":
        loss += abs(value) * config.regularization_beta

    if config.ie_over_seq:
        del upstream_activations_difference_ie_over_seq
    else:
        del upstream_activations_difference
    
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= batch_size
    graph.eap_scores = graph.eap_scores.cpu()
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
        device: str = "cuda:3"
        beta_1: float = 0.9
        beta_2: float = 0.9
        task: str = "sv"
        ie_over_seq: bool = False
        process_or_outcome: str = "outcome"
        ablation_method: str = "mean"
        smooth: bool = False
    
    config = Config()
    
    # model = HookedTransformer.from_pretrained(
    #     'gpt2-small',
    #     center_writing_weights=False,
    #     center_unembed=False,
    #     fold_ln=False,
    #     device="cuda:3"
    # )
    
    # model.set_use_hook_mlp_in(True)
    # model.set_use_split_qkv_input(True)
    # model.set_use_attn_result(True)
    
    
    def compare_sv_svd_old_old(model):
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
                )
        
            top_n = 50
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

            # key_nodes = common_nodes
            key_nodes = set([parent for parent, child in common_edges] + [child for parent, child in common_edges])
            key_edges = common_edges
            _, _ = graph.show(edges=top_edges, key_nodes=key_nodes, key_edges=key_edges, fname=f"eap_graph_sv_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_sv_svd_new")
            _, _ = graph.show(edges=top_edges_svd, key_nodes=key_nodes, key_edges=key_edges, fname=f"svd_graph_svd_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_sv_svd_new")
            break

    def compare_sv_svd_old(top_n=70):
        """
        Compare the sv circuit adn svd circuit
        """
        model = HookedTransformer.from_pretrained(
                'gpt2-small',
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                device="cuda:3"
            )
            
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        
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
                )
        
            top_edges = graph.top_edges(n=top_n, threshold=0, abs_scores=True)
            
            
            # get svd scores
            import jsonlines
            log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_5000-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
            top_edges_svd = None
            with jsonlines.open(log_path, "r") as f:
                for line in f:
                    if "edge_info" in line:
                        top_edges_svd = line["edge_info"]
                        break
            
            # log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_50-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
            # data = []
            # with jsonlines.open(log_path, "r") as f:
            #     for line in f:
            #         data.append(line)
            # top_edges_svd = data[-1]["edge_info"]
            
            top_edges_svd = sorted(top_edges_svd, key=lambda x: abs(x[-1]), reverse=True)[:top_n]
            
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
            # for i, edge in enumerate(top_edges_svd):
            #     up, down, score = edge
            #     if "head.0.5" in up:
            #         break
            # top_edges_svd.pop(i)
            
            # intersection
            common_nodes = list(sv_nodes.intersection(svd_nodes))
            common_edges = list(sv_edges.intersection(svd_edges))
            print("common edges:", common_edges)

            # key_nodes = common_nodes
            key_nodes = set([parent for parent, child in common_edges] + [child for parent, child in common_edges])
            key_edges = common_edges
            _, _ = graph.show(edges=top_edges, key_nodes=key_nodes, key_edges=key_edges, fname=f"eap_graph_sv_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_sv_svd_new")
            _, _ = graph.show(edges=top_edges_svd, key_nodes=key_nodes, key_edges=key_edges, fname=f"svd_graph_svd_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_sv_svd_new")
            break  
    
    def compare_sv_svd(top_n=70):
        """
        Compare the sv circuit adn svd circuit
        """
        model = HookedTransformer.from_pretrained(
                'gpt2-small',
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                device="cuda:0"
            )
            
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
        graph.reset_scores()
        patching_metric_fn = partial(patching_metric, logit_diff_func=avg_logit_diff_sv)
    
        train_dataset = SVDataset("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000.jsonl", model.tokenizer)
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
                )
        
            top_edges = graph.top_edges(n=top_n, threshold=0, abs_scores=True, prune_method="top_nodes")
            break
            
            # get svd scores
            # import jsonlines
            # log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{SGD}-warm_up_100-top_n_5000-threshold_0-random_ratio_0-ablation_{mean}-loss_weighted(p_1)/log.jsonl"
            # top_edges_svd = None
            # with jsonlines.open(log_path, "r") as f:
            #     for line in f:
            #         if "edge_info" in line:
            #             top_edges_svd = line["edge_info"]
            #             break
            
            # top_edges_svd = sorted(top_edges_svd, key=lambda x: abs(x[-1]), reverse=True)[:top_n]
        del model
        
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import load_model_from_ckpt
        cfg_path = "/home/lyy/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json"
        ckpt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/model-steps_1499_epoch_3.pt"
        model = load_model_from_ckpt("gpt2-small",
                                    ckpt_path,
                                    cfg_path,)
        
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        model.to("cuda:2")
        
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=sv_collate_fn)
    
        for batch in dataloader:
            graph, _, _ = EAP_ablation(
                model,
                graph=graph,
                batch=batch,
                metric=patching_metric_fn,
                config=config,
                )
        
            top_edges_svd = graph.top_edges(n=top_n, threshold=0, abs_scores=True, prune_method="top_nodes")
            break
        
        # ==========================================================================
        # get log scores
        for i in range(len(top_edges_svd)):
            top_edges_svd[i] = [top_edges_svd[i][0], top_edges_svd[i][1], np.log(top_edges_svd[i][2] + 1)]
        # for edge in top_edges_svd:
        #     edge[-1] = np.log(edge[-1] + 1)
    
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
        # for i, edge in enumerate(top_edges_svd):
        #     up, down, score = edge
        #     if "head.0.5" in up:
        #         break
        # top_edges_svd.pop(i)
        
        # intersection
        common_nodes = list(sv_nodes.intersection(svd_nodes))
        common_edges = list(sv_edges.intersection(svd_edges))
        print("common edges:", common_edges)

        key_nodes = common_nodes
        key_edges = common_edges
        # key_nodes = set([parent for parent, child in common_edges] + [child for parent, child in common_edges])
        
        _, _ = graph.show(edges=top_edges, key_nodes=key_nodes, key_edges=key_edges, fname=f"eap_graph_sv_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_lyy")
        _, _ = graph.show(edges=top_edges_svd, key_nodes=key_nodes, key_edges=key_edges, fname=f"eap_graph_svd_topn_{top_n}_1.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/figures/graph_lyy")
    
    def get_sv_graph(top_n):
        
        model = HookedTransformer.from_pretrained(
                'gpt2-small',
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                device="cuda:0"
            )
            
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        
        # upstream_nodes=["head"]
        # downstream_nodes=["head"]
        
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
                )
        
            top_edges = graph.top_edges(n=top_n, threshold=0, abs_scores=True, prune_method="top_nodes")
            break
        
        top_edges = sorted(top_edges, key=lambda x: x[-1], reverse=True)[:]
        
        _, edges = graph.show(edges=top_edges, fname=f"eap_graph_sv_topn_{top_n}_new.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims/new")

        # edge_info_path = os.path.join("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims", "sv_topn_50_sel_35_edge_info.jsonl")
        # import jsonlines
        # with jsonlines.open(edge_info_path, "w") as f:
        #     f.write({"edges": edges, "name": "eap_graph_sv_topn_50"})
    
    def get_svd_graph(top_n):
        
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import load_model_from_ckpt
        cfg_path = "/home/lyy/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json"
        ckpt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/model-steps_1499_epoch_3.pt"
        model = load_model_from_ckpt("gpt2-small",
                                    ckpt_path,
                                    cfg_path,)
        
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        model.to("cuda:1")
        
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
            graph, _, _ = EAP_ablation(
                model,
                graph=graph,
                batch=batch,
                metric=patching_metric_fn,
                config=config,
                )
        
            top_edges = graph.top_edges(n=top_n, threshold=0, abs_scores=True, prune_method="top_nodes")
            break
        
        top_edges = sorted(top_edges, key=lambda x: x[-1], reverse=True)[:]
        
        _, edges = graph.show(edges=top_edges, fname=f"eap_graph_svd_topn_{top_n}_new.pdf", fdir="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims/new")
    
    # get_sv_graph(top_n=100)
    get_svd_graph(top_n=100)
    # compare_sv_svd(top_n=70)        