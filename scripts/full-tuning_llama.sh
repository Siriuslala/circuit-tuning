cd ..


# single gpu ==============================================
# meta-llama/Llama-3.2-1B-Instruct / llama3.2-1b-it
N_DEVICES=1
DEVICE_ID=6
torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 44144  llama_full_tuning.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --model_path "" \
    --task "bias" \
    --device "cuda:$DEVICE_ID" \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_devices_per_node $N_DEVICES \
    --lr 1.5e-5 \
    --max_train_step 0 \


# FSDP ==============================================
# N_DEVICES=1
# export CUDA_VISIBLE_DEVICES=1
# torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 44144  llama_full_tuning.py \
#     --model_name "meta-llama/Llama-3.2-1B-Instruct" \
#     --model_path "" \
#     --task "bias" \
#     --batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --num_devices_per_node $N_DEVICES \
#     --lr 3e-5 \
#     --max_train_step 0 \
#     --enable_fsdp



# cd ../..
# python peft.py 2>&1 | tee $SAVE_DIR/train.log