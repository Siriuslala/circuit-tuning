"""
Utility functions for loading and processing models
"""

import torch
import torch.distributed.checkpoint as dist_cp

from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.utils as utils
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Config, LlamaForCausalLM, LlamaConfig

import numpy as np

from typing import Optional, Literal, Union, cast
import einops
import json
import jsonlines
import logging


DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def load_model_from_ckpt(
    official_model_name: str,
    ckpt_path: str,
    cfg_path: str,
    fold_ln: bool = False,
    center_writing_weights: bool = False,
    center_unembed: bool = False,
    refactor_factored_attn_matrices: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    n_devices: int = 1,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    move_to_device: bool = False,
    fold_value_biases: bool = False,
    default_prepend_bos: bool = False,
    default_padding_side: Literal["left", "right"] = "right",
    dtype="float32",
    first_n_layers: Optional[int] = None,
    **from_pretrained_kwargs,
) -> "HookedTransformer":
    """
    The function is modified from the original method "from_pretrained" in transformer_lens.HookedTransformer
    """

    if isinstance(dtype, str):
        # Convert from string to a torch dtype
        dtype = DTYPE_FROM_STRING[dtype]
    if "torch_dtype" in from_pretrained_kwargs:
        # For backwards compatibility with the previous way to do low precision loading
        # This should maybe check the user did not explicitly set dtype *and* torch_dtype
        dtype = from_pretrained_kwargs["torch_dtype"]

    if (
        (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
        or dtype == torch.float16
    ) and device in ["cpu", None]:
        logging.warning("float16 models may not work on CPU. Consider using a GPU or bfloat16.")

    # Load the config into an HookedTransformerConfig object.
    # cfg = get_model_config(
    #     official_model_name,
    #     cfg_path=cfg_path,
    #     fold_ln=fold_ln,
    #     device=device,
    #     n_devices=n_devices,
    #     default_prepend_bos=default_prepend_bos,
    #     dtype=dtype,
    #     first_n_layers=first_n_layers,
    #     **from_pretrained_kwargs,
    # )
    cfg = get_pretrained_model_config(
        official_model_name,
        cfg_path=cfg_path,
        fold_ln=fold_ln,
        device=device,
        n_devices=n_devices,
        default_prepend_bos=default_prepend_bos,
        dtype=dtype,
        first_n_layers=first_n_layers,
        **from_pretrained_kwargs,
    )
    
    # Get the tokenizer: no worries. (see "__init__" in HookedTransformer)
    
    # Create the HookedTransformer object
    split_params = True if "Circuit" in ckpt_path else False
    model = HookedTransformer(
        cfg,
        tokenizer,
        move_to_device=False,
        default_padding_side=default_padding_side,
        split_params=split_params,
    )
    
    # Get the state dict of the model
    if "fsdp" in ckpt_path.lower():
        print(f"loading model from model path: {ckpt_path} ")
        state_dict = {
            "model": model.state_dict()
        }
        dist_cp.load(state_dict=state_dict,
                    checkpoint_id=ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")  # about loading, see utils.download_file_from_hf

    # Convert to dtype
    # state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    model.load_and_process_state_dict(
        state_dict,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices,
    )

    if move_to_device:
        model.move_model_modules_to_device()

    ckpt_name = ckpt_path.split("/")[-1]
    print(f"Loaded pretrained model {ckpt_name} into HookedTransformer")

    return model


def load_model(model_name, ckpt_path, cfg_path, hf_model=None, tokenizer=None, device=None):
    """
    Move models from other formats to HookedTransformer.
    For "official models"(supported by TransformerLens), just use `HookedTransformer.from_pretrained`.
    For other models, use `load_model_from_ckpt`.
    """
    if hf_model:
        assert model_name is not None, "Please set your model name"
        assert tokenizer is not None, "Please set your tokenizer when using hf_model"
        
        if "llama" in model_name:
            # tokenizer.pad_token = tokenizer.eos_token
            
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # hf_model.resize_token_embeddings(len(tokenizer))
            # print("add pad token [PAD]")
            
        # model = HookedTransformer.from_pretrained(
        #     model_name=model_name,
        #     hf_model=hf_model,
        #     tokenizer=tokenizer,
        #     center_writing_weights=False,
        #     center_unembed=False,
        #     fold_ln=False,
        #     fold_value_biases=False,  # ?
        #     device=device,
        # )
            pass
        hf_model.to(device)
        
        return hf_model
        
    elif not ckpt_path:
        assert model_name is not None, "Please set your model name"
        model = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            fold_value_biases=False,
            device=device
        )
            
    else:
        model = load_model_from_ckpt(
            official_model_name=model_name,
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device=device
        )
    
    return model

def hookdedTF_to_TF(model_name, hookedTF_model, device="cuda:3"):
    """
    Move a HookedTransformer-format model to a HuggingFace model.
    """
    state_dict = {}
    model = None
    if "gpt2" in model_name.lower():
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModel(config)
        state_dict = format_hookedTF_to_TF_gpt2(hookedTF_model, config)
    elif "llama" in model_name.lower():
        if "1b" in model_name.lower():
            model_path = "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct"
        elif "3b" in model_name.lower():
            model_path = "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-3B-instruct"
        elif "8b" in model_name.lower():
            model_path = "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct"
        config = LlamaConfig.from_pretrained(model_path)
        model = LlamaForCausalLM(config)
        state_dict = format_hookedTF_to_TF_llama_split(hookedTF_model, config)
    else:
        raise ValueError("Unsupported model name")
    
    state_dict_keys = list(state_dict.keys())
    for key in state_dict_keys:
        model.load_state_dict({key: state_dict[key]}, strict=False)
        del state_dict[key]
        
    print(f"Loaded model {model_name} from HookedTransformer into {type(model)}")
    
    return model

def format_hookedTF_to_TF_gpt2(hookedTF_model, config: LlamaConfig):
    """
    Used for moving state_dict of gpt2 from HookedTransformer to transformers (GPT2LMHeadModel)
    This is actually a reverse version of `transformer_lens/pretrained/weight_conversions/gpt2.py`
    
    你是一个很厉害的程序员。现在我有两个版本的模型，一个是HookedTransformer版本，一个是官方版本 GPT2LMHeadModel。现在我有将官方版本模型 GPT2LMHeadModel 的state_dict转为hookedTransformer版本的函数：convert_gpt2_weights，请你帮我写一个将HookedTransformer版本的模型的state_dict转为官方版本 GPT2LMHeadModel 的方法。我会给你convert_gpt2_weights函数以及官方版本模型的架构：
    
    GPT2-hookedTF:
    GPT2-TF:
    
    """
    
    gpt2_model = GPT2LMHeadModel(config)
    gpt2_state_dict = gpt2_model.state_dict()

    gpt2_state_dict['transformer.wte.weight'] = hookedTF_model.W_E
    gpt2_state_dict['transformer.wpe.weight'] = hookedTF_model.W_pos

    W_Q = hookedTF_model.W_Q  # torch.stack([block.attn.W_Q for block in self.blocks], dim=0) ~ (n_layers, n_heads, d_head, d_model)
    W_K = hookedTF_model.W_K  # model.W_K -> torch.repeat_interleave(self._b_K, dim=0, repeats=self.repeat_kv_heads)
    W_V = hookedTF_model.W_V
    W_O = hookedTF_model.W_O
    b_Q = hookedTF_model.b_Q
    b_K = hookedTF_model.b_K
    b_V = hookedTF_model.b_V
    b_O = hookedTF_model.b_O
    
    W_in = hookedTF_model.W_in
    W_out = hookedTF_model.W_out
    b_in = hookedTF_model.b_in
    b_out = hookedTF_model.b_out
    Ln1_w = torch.stack([block.ln1.w for block in hookedTF_model.blocks], dim=0)
    Ln1_b = torch.stack([block.ln1.b for block in hookedTF_model.blocks], dim=0)
    Ln2_w = torch.stack([block.ln2.w for block in hookedTF_model.blocks], dim=0)
    Ln2_b = torch.stack([block.ln2.b for block in hookedTF_model.blocks], dim=0)
    
    for l in range(config.num_hidden_layers):
        gpt2_state_dict[f'transformer.h.{l}.ln_1.weight'] = Ln1_w[l]
        gpt2_state_dict[f'transformer.h.{l}.ln_1.bias'] = Ln1_b[l]
        
        W_Q_l = einops.rearrange(W_Q[l], 'i m h -> m (i h)', i=config.num_attention_heads)
        W_K_l = einops.rearrange(W_K[l], 'i m h -> m (i h)', i=config.num_attention_heads)
        W_V_l = einops.rearrange(W_V[l], 'i m h -> m (i h)', i=config.num_attention_heads)
        W = torch.cat([W_Q_l, W_K_l, W_V_l], dim=1)
        gpt2_state_dict[f'transformer.h.{l}.attn.c_attn.weight'] = W

        b_Q_l = b_Q[l]  # (index, head)
        b_K_l = b_K[l]
        b_V_l = b_V[l]
        b = torch.stack([b_Q_l, b_K_l, b_V_l], dim=0)
        b = einops.rearrange(
            b, 
            'qkv index head -> (qkv index head)', 
            qkv=3, 
            index=config.num_attention_heads)
        gpt2_state_dict[f'transformer.h.{l}.attn.c_attn.bias'] = b

        W_O_l = einops.rearrange(W_O[l], 'i h m -> m (i h)', i=config.num_attention_heads)
        gpt2_state_dict[f'transformer.h.{l}.attn.c_proj.weight'] = W_O_l
        gpt2_state_dict[f'transformer.h.{l}.attn.c_proj.bias'] = b_O[l]

        gpt2_state_dict[f'transformer.h.{l}.ln_2.weight'] = Ln2_w[l]
        gpt2_state_dict[f'transformer.h.{l}.ln_2.bias'] = Ln2_b[l]

        gpt2_state_dict[f'transformer.h.{l}.mlp.c_fc.weight'] = W_in[l]
        gpt2_state_dict[f'transformer.h.{l}.mlp.c_fc.bias'] = b_in[l]

        gpt2_state_dict[f'transformer.h.{l}.mlp.c_proj.weight'] = W_out[l]
        gpt2_state_dict[f'transformer.h.{l}.mlp.c_proj.bias'] = b_out[l]

    gpt2_state_dict['lm_head.weight'] = hookedTF_model.W_U.T

    gpt2_state_dict['transformer.ln_f.weight'] = hookedTF_model.ln_final.w
    gpt2_state_dict['transformer.ln_f.bias'] = hookedTF_model.ln_final.b

    return gpt2_state_dict

def format_hookedTF_to_TF_llama(hookedTF_model, config: LlamaConfig):
    """
    Used for moving state_dict of llama from HookedTransformer to transformers (LlamaForCausalLM)
    This is actually a reverse version of `transformer_lens/pretrained/weight_conversions/llama.py`
    
    你是一个很厉害的程序员。现在我有两个版本的模型，一个是HookedTransformer版本，一个是官方版本 LlamaForCausalLM。现在我有将官方版本模型 LlamaForCausalLM的state_dict转为hookedTransformer版本的函数：convert_llama_weights，请你帮我写一个将HookedTransformer版本的模型的state_dict转为官方版本 LlamaForCausalLM的方法。我会给你convert_llama_weights函数以及官方版本模型的架构：
    
    """
    
    llama_model = LlamaForCausalLM(config)
    llama_state_dict = llama_model.state_dict()

    # 复制HookedTransformer的权重到Llama模型结构中
    llama_state_dict['model.embed_tokens.weight'] = hookedTF_model.W_E

    W_Q = hookedTF_model.W_Q  # torch.stack([block.attn.W_Q for block in self.blocks], dim=0) ~ (n_layers, n_heads, d_head, d_model)
    W_K = torch.stack([block.attn._W_K for block in hookedTF_model.blocks], dim=0)  # model.W_K -> torch.repeat_interleave(self._b_K, dim=0, repeats=self.repeat_kv_heads)
    W_V = torch.stack([block.attn._W_V for block in hookedTF_model.blocks], dim=0)
    W_O = hookedTF_model.W_O
    W_in = hookedTF_model.W_in
    W_gate = hookedTF_model.W_gate
    W_out = hookedTF_model.W_out
    Ln1 = torch.stack([block.ln1.w for block in hookedTF_model.blocks], dim=0)
    Ln2 = torch.stack([block.ln2.w for block in hookedTF_model.blocks], dim=0)
    
    for l in range(config.num_hidden_layers):
        llama_state_dict[f'model.layers.{l}.input_layernorm.weight'] = Ln1[l]
        llama_state_dict[f'model.layers.{l}.self_attn.q_proj.weight'] = einops.rearrange(
            W_Q[l], 'n m h -> (n h) m', n=config.num_attention_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.k_proj.weight'] = einops.rearrange(
            W_K[l], 'n m h -> (n h) m', n=config.num_key_value_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.v_proj.weight'] = einops.rearrange(
            W_V[l], 'n m h -> (n h) m', n=config.num_key_value_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.o_proj.weight'] = einops.rearrange(
            W_O[l], 'n h m -> m (n h)', n=config.num_attention_heads
        )
        llama_state_dict[f'model.layers.{l}.post_attention_layernorm.weight'] = Ln2[l]
        llama_state_dict[f'model.layers.{l}.mlp.up_proj.weight'] = W_in[l].T
        llama_state_dict[f'model.layers.{l}.mlp.gate_proj.weight'] = W_gate[l].T
        llama_state_dict[f'model.layers.{l}.mlp.down_proj.weight'] = W_out[l].T

    llama_state_dict['model.norm.weight'] = hookedTF_model.ln_final.w
    llama_state_dict['lm_head.weight'] = hookedTF_model.W_U.T

    return llama_state_dict

def format_hookedTF_to_TF_llama_split(hookedTF_model, config: LlamaConfig):
    """
    Used for moving state_dict of llama from HookedTransformer to transformers (LlamaForCausalLM)
    This is actually a reverse version of `transformer_lens/pretrained/weight_conversions/llama.py`
    
    你是一个很厉害的程序员。现在我有两个版本的模型，一个是HookedTransformer版本，一个是官方版本 LlamaForCausalLM。现在我有将官方版本模型 LlamaForCausalLM的state_dict转为hookedTransformer版本的函数：convert_llama_weights，请你帮我写一个将HookedTransformer版本的模型的state_dict转为官方版本 LlamaForCausalLM的方法。我会给你convert_llama_weights函数以及官方版本模型的架构：
    
    """
    
    llama_model = LlamaForCausalLM(config)
    llama_state_dict = llama_model.state_dict()

    # 复制HookedTransformer的权重到Llama模型结构中
    llama_state_dict['model.embed_tokens.weight'] = hookedTF_model.W_E

    W_Q = hookedTF_model.W_Q  # torch.stack([block.attn.W_Q for block in self.blocks], dim=0) ~ (n_layers, n_heads, d_head, d_model)
    W_K = torch.stack([block.attn.W_K_origin for block in hookedTF_model.blocks], dim=0)  # model.W_K -> torch.repeat_interleave(self._b_K, dim=0, repeats=self.repeat_kv_heads)
    W_V = torch.stack([block.attn.W_V_origin for block in hookedTF_model.blocks], dim=0)
    W_O = hookedTF_model.W_O
    W_in = hookedTF_model.W_in
    W_gate = hookedTF_model.W_gate
    W_out = hookedTF_model.W_out
    Ln1 = torch.stack([block.ln1.w for block in hookedTF_model.blocks], dim=0)
    # print("LN1", Ln1.shape)
    Ln2 = torch.stack([block.ln2.w for block in hookedTF_model.blocks], dim=0)
    
    for l in range(config.num_hidden_layers):
        llama_state_dict[f'model.layers.{l}.input_layernorm.weight'] = Ln1[l]
        llama_state_dict[f'model.layers.{l}.self_attn.q_proj.weight'] = einops.rearrange(
            W_Q[l], 'n m h -> (n h) m', n=config.num_attention_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.k_proj.weight'] = einops.rearrange(
            W_K[l], 'n m h -> (n h) m', n=config.num_key_value_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.v_proj.weight'] = einops.rearrange(
            W_V[l], 'n m h -> (n h) m', n=config.num_key_value_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.o_proj.weight'] = einops.rearrange(
            W_O[l], 'n h m -> m (n h)', n=config.num_attention_heads
        )
        llama_state_dict[f'model.layers.{l}.post_attention_layernorm.weight'] = Ln2[l]
        llama_state_dict[f'model.layers.{l}.mlp.up_proj.weight'] = W_in[l].T
        llama_state_dict[f'model.layers.{l}.mlp.gate_proj.weight'] = W_gate[l].T
        llama_state_dict[f'model.layers.{l}.mlp.down_proj.weight'] = W_out[l].T     

    llama_state_dict['model.norm.weight'] = hookedTF_model.ln_final.w
    llama_state_dict['lm_head.weight'] = hookedTF_model.W_U.T
    
    return llama_state_dict

def format_hookedTF_to_TF_llama_split_new(hookedTF_model, config: LlamaConfig):
    """
    Used for moving state_dict of llama from HookedTransformer to transformers (LlamaForCausalLM)
    This is actually a reverse version of `transformer_lens/pretrained/weight_conversions/llama.py`
    
    你是一个很厉害的程序员。现在我有两个版本的模型，一个是HookedTransformer版本，一个是官方版本 LlamaForCausalLM。现在我有将官方版本模型 LlamaForCausalLM的state_dict转为hookedTransformer版本的函数：convert_llama_weights，请你帮我写一个将HookedTransformer版本的模型的state_dict转为官方版本 LlamaForCausalLM的方法。我会给你convert_llama_weights函数以及官方版本模型的架构：
    
    """
    
    llama_model = LlamaForCausalLM(config)
    llama_state_dict = llama_model.state_dict()

    # 复制HookedTransformer的权重到Llama模型结构中
    llama_state_dict['model.embed_tokens.weight'] = hookedTF_model.W_E
    
    for l in range(config.num_hidden_layers):
        llama_state_dict[f'model.layers.{l}.input_layernorm.weight'] = torch.stack([block.ln1.w for block in hookedTF_model.blocks], dim=0)[l]
        llama_state_dict[f'model.layers.{l}.self_attn.q_proj.weight'] = einops.rearrange(
            hookedTF_model.W_Q[l], 'n m h -> (n h) m', n=config.num_attention_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.k_proj.weight'] = einops.rearrange(
            torch.stack([block.attn.W_K_origin for block in hookedTF_model.blocks], dim=0)[l], 'n m h -> (n h) m', n=config.num_key_value_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.v_proj.weight'] = einops.rearrange(
            torch.stack([block.attn.W_V_origin for block in hookedTF_model.blocks], dim=0)[l], 'n m h -> (n h) m', n=config.num_key_value_heads
        )
        llama_state_dict[f'model.layers.{l}.self_attn.o_proj.weight'] = einops.rearrange(
            hookedTF_model.W_O[l], 'n h m -> m (n h)', n=config.num_attention_heads
        )
        llama_state_dict[f'model.layers.{l}.post_attention_layernorm.weight'] = torch.stack([block.ln2.w for block in hookedTF_model.blocks], dim=0)[l]
        llama_state_dict[f'model.layers.{l}.mlp.up_proj.weight'] = hookedTF_model.W_in[l].T
        llama_state_dict[f'model.layers.{l}.mlp.gate_proj.weight'] = hookedTF_model.W_gate[l].T
        llama_state_dict[f'model.layers.{l}.mlp.down_proj.weight'] = hookedTF_model.W_out[l].T
        
        del hookedTF_model.blocks[l]  

    llama_state_dict['model.norm.weight'] = hookedTF_model.ln_final.w
    llama_state_dict['lm_head.weight'] = hookedTF_model.W_U.T
    
    del hookedTF_model

    return llama_state_dict


class TopNScheduler:
    """
    topn = start + delta_top_n
    """
    def __init__(self, start_val, end_val, warmup_steps, scheduler_type="linear"):
        self.start = start_val
        self.end = end_val
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.step_cnt = 0
    
    def step(self):
        
        if self.scheduler_type == "linear":
            rate = (self.end - self.start) / self.warmup_steps
            if rate > 0:
                top_n = int(min(self.start + rate * self.step_cnt, self.end))
            else:
                top_n = int(max(self.start + rate * self.step_cnt, self.end))
                
        elif self.scheduler_type == "cosine":
            amp = (self.start - self.end) / 2
            T = self.warmup_steps * 2
            if self.step_cnt < self.warmup_steps:
                top_n = int(amp * np.cos(2 * np.pi * self.step_cnt / T) + amp + self.end)
            else:
                top_n = self.end
        else:
            top_n = self.end
        
        self.step_cnt += 1
        
        return top_n
    

def print_rank_0(*args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

if __name__ == "__main__":
    
    # tf_to_hf("meta-llama/Llama-3.2-1B-instruct", device=None)
    
    def test_llama_format():
        
        hf_model = AutoModelForCausalLM.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
        
        hooked_model = HookedTransformer.from_pretrained(
            model_name='meta-llama/Llama-3.2-1B-instruct',
            hf_model=hf_model,
            tokenizer=tokenizer,
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda:0",
        )
        
        hookdedTF_to_TF("Llama-3.2-1B-instruct", hooked_model, device="cuda:0")
    
    def test_gpt2_format():
        
        hooked_model = HookedTransformer.from_pretrained(
            model_name='gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda:0",
        )
        
        hookdedTF_to_TF("GPT2", hooked_model, device="cuda:0")
   
    def test_math_dataset():
        
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
        path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/math_dataset/train.jsonl"
        data = []
        with jsonlines.open(path, "r") as f:
            for line in f:
                data.append(line["question"] + line["answer"])
        lens = []
        data_tokens = []
        for item in data:
            tokens = tokenizer.tokenize(item)
            lens.append(len(tokens))
            data_tokens.append(tokens)
        print(max(lens), min(lens), sum(lens) / len(lens))  # 440, 51, 158   # max_len = 512
        max_id = lens.index(max(lens))
        print(data[max_id])
        min_id = lens.index(min(lens))
        print(data[min_id])
        print(data_tokens[min_id])
        
        lens.pop(max_id)
        print(max(lens))
    
    def test_llama_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
        print(tokenizer.padding_side)
        tokenizer.pad_token = tokenizer.eos_token
        sample0 = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                  "labels": [-100, -100, -100, 4, 5, 6, 7, 8, 9, 10],
                  "question": "qn",
                  "answer": "answer"}
        sample1 = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                  "labels": [-100, -100, -100, 4, 5, 6, 7, 8, 9, 10],
                  "question": "qn",
                  "answer": "answer"}
        sample2 = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                  "labels": [-100, -100, -100, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  "question": "qn",
                  "answer": "answer"}
        batch = [sample0, sample1, sample2]
        res = tokenizer.pad(batch, return_tensors="pt")
        print(res)
    
    
    def test_truncate():
        tokenizer = AutoTokenizer.from_pretrained("/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
        tokenizer.pad_token = tokenizer.eos_token
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), 
                 "labels": torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8, 9, 10],
                                         [-100, -100, -100, -100, 5, 6, 7, 8, 9, 10],
                                         [-100, -100, -100, 4, 5, 6, 7, 8, 9, 10]]),}
        labels = batch["labels"]
        # keep prefix (corresponds to -100 in labels) and leave out the rest
        mask = batch["labels"] == -100
        # using mask to truncate input_ids, while keep the shape of input_ids
        batch["input_ids"] = batch["input_ids"].masked_fill(~mask, tokenizer.pad_token_id)
        print(batch["input_ids"])
        # move the unmasked elemnets to the right
        # max_len = batch["input_ids"].shape[1]
        
        for i in range(len(batch)):
            qn_end = torch.nonzero(batch["labels"][i] == -100)[-1]
            batch["input_ids"][i] = torch.cat([batch["input_ids"][i][qn_end+1:], batch["input_ids"][i][:qn_end+1]])
        print(batch["input_ids"])
        
    # test_gpt2_format()
    # test_truncate()
    
 