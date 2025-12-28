"""
Full-tuning for Llamas
"""

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_recipes.configs import train_config as TRAIN_CONFIG, fsdp_config as FSDP_CONFIG
from llama_recipes.utils.config_utils import (
    get_dataloader_kwargs,
    check_fsdp_config,
)
from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)
from llama_recipes.utils.train_utils import (
    train, 
    print_model_size, 
    setup, 
    clear_gpu_cache, 
    setup_environ_flags,
    freeze_transformer_layers,
    get_policies,
)
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from utils import print_rank_0

import numpy as np
from accelerate.utils import is_xpu_available
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

import os
import random
import argparse

# ========================== Args ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct")
parser.add_argument("--model_name", type=str, default="llama-3.2-1B-it")
parser.add_argument("--task", type=str, default="bias")
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_train_step", type=int, default=500)
parser.add_argument("--num_devices_per_node", type=int, default=2)
args = parser.parse_args()

# Use args for my own appetite!
args = {"model_path": args.model_path,
        "model_name": args.model_name, 
        "task": args.task,
        "num_epochs": 1, 
        "gradient_accumulation_steps": args.gradient_accumulation_steps, 
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": 100,
        "max_train_step": args.max_train_step,
        "context_length": 512,
        "precision": "bf16",
        "device": "cuda:3",
        "num_devices_per_node": args.num_devices_per_node,
}

bsz = args['batch_size'] * args['gradient_accumulation_steps'] * args['num_devices_per_node']

args["ckpt_name"] = f"{args['model_name']}-{args['task']}-epochs_{args['num_epochs']}-bsz_{bsz}-lr_{args['lr']}-Opt_SGD-full-precision_{args['precision']}-1"
args["output_dir"] = os.path.join(f"/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-{args['task']}", args["ckpt_name"])
args["log_path"] = os.path.join(args["output_dir"], "log.jsonl")

train_config = TRAIN_CONFIG()
train_config.model_name = args["model_path"]
train_config.use_fp16 = not args['precision'] == "bf16"
train_config.num_epochs = args["num_epochs"]
train_config.max_train_step = args["max_train_step"]
train_config.use_peft = False
train_config.lr = args["lr"]
train_config.batch_size_training = args["batch_size"]
train_config.gradient_accumulation_steps = args["gradient_accumulation_steps"]
train_config.dataset = "custom_dataset"
train_config.context_length = 512  # used for packing
train_config.batching_strategy = "padding"
train_config.run_validation = False
train_config.val_batch_size = 16
train_config.output_dir = args["output_dir"]
train_config.use_fast_kernels = True
train_config.enable_fsdp = True
train_config.quantization = None
train_config.dist_checkpoint_root_folder = args["output_dir"]
train_config.dist_checkpoint_folder = "fsdp"
train_config.use_wandb = True
train_config.save_metrics = True
train_config.seed = 14

fsdp_config = FSDP_CONFIG()
fsdp_config.pure_bf16 = True

torch.manual_seed(train_config.seed)
random.seed(train_config.seed)
np.random.seed(train_config.seed)

# ========================== Pre ==========================
if train_config.enable_fsdp:
    setup()
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

if torch.distributed.is_initialized():
    if is_xpu_available():
        torch.xpu.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)
    setup_environ_flags(rank)


# ========================== Load the model ==========================
from transformers import BitsAndBytesConfig
# quant_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

use_cache = False if train_config.enable_fsdp else None
model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.bfloat16,
        )

print_rank_0("Model loaded")
print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

if (
        train_config.enable_fsdp
        and fsdp_config.pure_bf16
        and not train_config.quantization
    ):
        model.to(torch.bfloat16)

# ========================== Prepare parallel training ==========================
hsdp_device_mesh_plan = None
if (
    fsdp_config.hsdp
    and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
):
    hsdp_device_mesh_plan = hsdp_device_mesh(
        replica_group_size=fsdp_config.replica_group_size,
        sharding_group_size=fsdp_config.sharding_group_size,
    )
    print_rank_0("HSDP device mesh is ready")

# setting up FSDP if enable_fsdp is enabled
if train_config.enable_fsdp:
    check_fsdp_config(fsdp_config)

    if not train_config.use_peft and train_config.freeze_layers:
        freeze_transformer_layers(model, train_config.num_freeze_layers)

    mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
    
    # Create the FSDP wrapper for LlamaDecoderLayer in text models
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
    device_id = 0
    if is_xpu_available():
        device_id = torch.xpu.current_device()
    elif torch.cuda.is_available():
        device_id = torch.cuda.current_device()
    model = FSDP(
        model,
        auto_wrap_policy=(
            my_auto_wrapping_policy if train_config.use_peft else wrapping_policy
        ),
        cpu_offload=(
            CPUOffload(offload_params=True)
            if fsdp_config.fsdp_cpu_offload
            else None
        ),
        mixed_precision=(
            mixed_precision_policy if not fsdp_config.pure_bf16 else None
        ),
        sharding_strategy=fsdp_config.sharding_strategy,
        device_mesh=hsdp_device_mesh_plan,
        device_id=device_id,
        limit_all_gathers=True,
        sync_module_states=train_config.low_cpu_fsdp,
        param_init_fn=(
            (
                lambda module: module.to_empty(
                    device=torch.device("cuda"), recurse=False
                )
            )
            if train_config.low_cpu_fsdp and rank != 0
            else None
        ),
    )
    if fsdp_config.fsdp_activation_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        apply_fsdp_checkpointing(model)
elif not train_config.quantization and not train_config.enable_fsdp:
    if is_xpu_available():
        model.to("xpu:0")
    elif torch.cuda.is_available():
        model.to(args["device"])
            
            
# ========================== Load the dataset ==========================
from llama_recipes.configs.datasets import custom_dataset
from llama_recipes.data.concatenator import ConcatDataset
from dataclasses import dataclass
# two ways to load dataset
# (1) write a function returning the dataset in a .py file which can be given to the command line tool.
# (2) change the source code: configs/datasets.py.

dataset_config = custom_dataset()
if args["task"] == "logic":
    dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/logic_dataset.py"
elif args["task"] == "math":
    dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/math_dataset.py"
elif args["task"] == "bias":
    dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/bias_dataset.py"
elif args["task"] == "reading":
    dataset_config.file = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad_dataset.py"
else:
    raise ValueError("Invalid task")
dataset_config.train_split = "train"
dataset_config.test_split = "test"


dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
print_rank_0(f"--> Training Set Length = {len(dataset_train)}")

# dataset_val = get_preprocessed_dataset(
#     tokenizer,
#     dataset_config,
#     split="test",
# )
# print_rank_0(f"--> Validation Set Length = {len(dataset_val)}")

if train_config.batching_strategy == "packing":
    dataset_train = ConcatDataset(
        dataset_train, chunk_size=train_config.context_length
    )

train_dl_kwargs = get_dataloader_kwargs(
    train_config, dataset_train, tokenizer, "train"
)
custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)
if custom_data_collator:
    print_rank_0("custom_data_collator is used")
    train_dl_kwargs["collate_fn"] = custom_data_collator
    
# Create DataLoaders for the training and validation dataset
train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    num_workers=train_config.num_workers_dataloader,
    pin_memory=True,
    **train_dl_kwargs,
)
print_rank_0(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

eval_dataloader = None
# if train_config.run_validation:
#     if train_config.batching_strategy == "packing":
#         dataset_val = ConcatDataset(
#             dataset_val, chunk_size=train_config.context_length
#         )

#     val_dl_kwargs = get_dataloader_kwargs(
#         train_config, dataset_val, tokenizer, "val"
#     )
#     if custom_data_collator:
#         val_dl_kwargs["collate_fn"] = custom_data_collator

#     eval_dataloader = torch.utils.data.DataLoader(
#         dataset_val,
#         num_workers=train_config.num_workers_dataloader,
#         pin_memory=True,
#         **val_dl_kwargs,
#     )
#     print_rank_0(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")


# ========================== Train the model ==========================
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb

model.train()

# optimizer = optim.AdamW(
#             model.parameters(),
#             lr=train_config.lr,
#             weight_decay=train_config.weight_decay,
#         )
# mini-batch SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
    momentum=0.9,
)

scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
# scheduler = optim.lr_scheduler.LambdaLR(
#             optimizer,
#             lr_lambda=lambda step: min(1.0, step / args.warmup_steps),
#         )

if rank == 0:
    wandb.login(key="f33d04e8bfbd4c33be09b9d9f41445e390346cf6")
    print_rank_0("circuit-tuning")
    wandb_run = wandb.init(project=f"circuit-tuning-{args['task']}", name=args["ckpt_name"])
else:
    wandb_run = None

# Start the training process
results = train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    fsdp_config=fsdp_config if train_config.enable_fsdp else None,
    local_rank=local_rank if train_config.enable_fsdp else None,
    rank=rank if train_config.enable_fsdp else None,
    wandb_run=wandb_run,
    device=args["device"],
    added_eval=None,  # evaluate_math_llama_recipe
    log_path=args["log_path"],
)

if not train_config.enable_fsdp or rank == 0:
        [print_rank_0(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

# model.save_pretrained(train_config.output_dir)


# model.eval()
# with torch.inference_mode():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))