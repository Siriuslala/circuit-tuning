"""Hooked Transformer Gated MLP Component.

This module contains all the component :class:`GatedMLP`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.addmm import batch_addmm

import einops

if is_bitsandbytes_available():
    pass


class GatedMLP(CanBeUsedAsMLP):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)
        self.select_activation_function()
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.d_mlp, dtype=self.cfg.dtype))
        self.W_out = nn.Parameter(torch.empty(self.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype))
        self.W_gate = nn.Parameter(torch.empty(self.cfg.d_model, self.d_mlp, dtype=self.cfg.dtype))

        self.b_in = nn.Parameter(torch.zeros(self.d_mlp, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            torch.matmul(x, self.W_gate)  # batch pos d_model, d_model d_mlp -> batch pos d_mlp
        )  # [batch, pos, d_mlp]

        if (
            self.cfg.is_layer_norm_activation()
            and self.hook_mid is not None
            and self.ln is not None
        ):
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        else:
            pre_linear = self.hook_pre_linear(
                torch.matmul(x, self.W_in)  # batch pos d_model, d_model d_mlp -> batch pos d_mlp
            )

            post_act = self.hook_post(
                (self.act_fn(pre_act) * pre_linear) + self.b_in
            )  # [batch, pos, d_mlp]

        return batch_addmm(self.b_out, self.W_out, post_act)


class GatedMLPSplit(CanBeUsedAsMLP):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)
        self.select_activation_function()
        self.d_mlp_head = 64
        self.n_mlp_head = self.d_mlp // self.d_mlp_head  # 128
        
        for i in range(self.n_mlp_head):
            param = nn.Parameter(
                torch.empty(
                    self.cfg.d_model,
                    self.d_mlp_head,
                    dtype=cfg.dtype
                )
            )
            setattr(self, f'W_in_{i}', param)
        
        for i in range(self.n_mlp_head):
            param = nn.Parameter(
                torch.empty(
                    self.d_mlp_head,
                    self.cfg.d_model,
                    dtype=cfg.dtype
                )
            )
            setattr(self, f'W_out_{i}', param)
            
        for i in range(self.n_mlp_head):
            param = nn.Parameter(
                torch.empty(
                    self.cfg.d_model,
                    self.d_mlp_head,
                    dtype=cfg.dtype
                )
            )
            setattr(self, f'W_gate_{i}', param)

        self.b_in =  nn.Parameter(torch.zeros(self.d_mlp, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        
        x = einops.repeat(
            x,
            'batch pos d_model -> batch pos n_mlp_head d_model',
            n_mlp_head=self.n_mlp_head
        ).clone()
        
        pre_act = self.hook_pre(
            einops.einsum(
                x,
                torch.stack([getattr(self, f'W_gate_{i}') for i in range(self.n_mlp_head)], dim=0),
                "batch pos n_mlp_head d_model, n_mlp_head d_model d_mlp_head -> batch pos n_mlp_head d_mlp_head"    
            )
        )
                
        if (
            self.cfg.is_layer_norm_activation()
            and self.hook_mid is not None
            and self.ln is not None
        ):
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
            
        else:
            pre_linear = self.hook_pre_linear(
                einops.einsum(
                    x,
                    torch.stack([getattr(self, f'W_in_{i}') for i in range(self.n_mlp_head)], dim=0),
                    "batch pos n_mlp_head d_model, n_mlp_head d_model d_mlp_head -> batch pos n_mlp_head d_mlp_head",
                      
                )
            )           
            # pre_act = pre_act.reshape(pre_act.shape[0], pre_act.shape[1], self.d_mlp)  # batch pos d_mlp
            # pre_linear = pre_linear.reshape(pre_linear.shape[0], pre_linear.shape[1], self.d_mlp)
                    
            post_act = self.hook_post(
                self.act_fn(pre_act) * pre_linear
            )  # [batch, pos, d_mlp] / batch pos n_mlp_head d_mlp_head
        
        out = einops.einsum(
            post_act,
            torch.stack([getattr(self, f'W_out_{i}') for i in range(self.n_mlp_head)], dim=0),  # [d_mlp, d_model]
            "batch pos n_mlp_head d_mlp_head, n_mlp_head d_mlp_head d_model -> batch pos n_mlp_head d_model"
        )

        # return batch_addmm(self.b_out,
        #                    torch.cat([getattr(self, f'W_out_{i}') for i in range(self.n_mlp_head)], dim=0),  # [d_mlp, d_model] 
        #                    post_act)  # batch pos d_model
        
        return out

    @property
    def W_in(self):
        return torch.cat([getattr(self, f'W_in_{i}') for i in range(self.n_mlp_head)], dim=-1)

    @property
    def W_gate(self):
        return torch.cat([getattr(self, f'W_gate_{i}') for i in range(self.n_mlp_head)], dim=-1)
    
    @property
    def W_out(self):
        return torch.cat([getattr(self, f'W_out_{i}') for i in range(self.n_mlp_head)], dim=0)