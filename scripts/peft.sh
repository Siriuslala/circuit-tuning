cd ..

# meta-llama/Llama-3.2-1B-Instruct / llama3.2-1b-it
N_DEVICES=1
export CUDA_VISIBLE_DEVICES=7
torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 44113  peft_llama.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --model_path "" \
    --task "bias" \
    --num_epochs 1 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_devices_per_node $N_DEVICES \
    --lr 1e-4 \
    --lora_r 32 \
    --lora_alpha 32 \
    --max_train_step 0
    


# cd ../..
# python peft.py 2>&1 | tee $SAVE_DIR/train.log