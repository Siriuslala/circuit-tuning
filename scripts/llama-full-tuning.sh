


# llama_recipes.finetuning needs to be modified to support my appetite, which involves adding lots of new params, so I don't use it directly. Instead, I use the following script to run the finetuning process.

# python -m llama_recipes.finetuning \
#     --model_name "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct" \
#     --num_epochs 2 \
#     --lr 1e-4 \
#     --batch_size_training 4 \
#     --gradient_accumulation_steps 4 \
#     --use_peft \
#     --peft_method lora \
#     --dataset "custom_dataset" \
#     --custom_dataset.file "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/gms8k_dataset.py" \
#     --context_length 512 \
#     --output_dir "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math" \
#     --use_wandb \
#     --save_metrics \
#     --use_fast_kernels \

cd /raid_sdd/lyy/Interpretability/lyy/circuit-tuning

N_DEVICES=2
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 44141  llama_full_tuning.py \
    --model_name ""llama3.1-8b-it"" \
    --model_path "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct" \
    --task "logic" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_devices_per_node $N_DEVICES \
    --lr 1e-5 \
    --max_train_step 500 \


# N_DEVICES=2
# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 44141  llama_full_tuning.py \
#     --model_name ""llama3.2-1b-it"" \
#     --model_path "/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct" \
#     --task "reading" \
#     --num_epochs 1 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --num_devices_per_node $N_DEVICES \
#     --lr 1e-5 \
#     --max_train_step 500 \



# cd /raid_sdd/lyy/Interpretability/lyy/circuit-tuning
# python peft.py 2>&1 | tee $SAVE_DIR/train.log