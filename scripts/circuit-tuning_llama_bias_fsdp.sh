source exp_info.sh

cd ..

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-3B-Instruct"
TASK="bias"

NUM_EPOCHS=2
LR=5e-5  #
BATCH_SIZE=1
ACCUMULATION_STEPS=8
N_DEVICES=2

OPTIMIZER_NAME="SGD"

BSZ=$(expr $BATCH_SIZE \* $ACCUMULATION_STEPS \* $N_DEVICES)

PRUNE_METHOD="top_edges"
TOP_N=10000 #
THRESHOLD=0
RAMDOM_RATIO=0
RANDN_KEY=0
TOPN_START=0 #
TOPN_WARMUP=0  # 
TOPN_TYPE="cosine"
PROCESS_OR_OUTCOME="outcome"
ABLATION_METHOD="mean"
OBJECTIVE="standard"
LOSS_WEIGHT_P=1
REGULARIZATION_BETA=2  # 

WANDB_PROJECT_NAME="circuit-tuning-$TASK"
PRINT_EVERY=10

# ckpt_name
IFS='/' read -ra PARTS <<< "$MODEL_NAME"
MODEL=${PARTS[-1]}

CKPT_NAME="Circuit-$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-top_n_$TOP_N-topn_start_$TOPN_START-warmup_$TOPN_WARMUP-$TOPN_TYPE-reg_${REGULARIZATION_BETA}"

SAVE_DIR="$WORK_DIR/checkpoints-bias/${CKPT_NAME}"

mkdir -p $SAVE_DIR

device_id=3
export CUDA_VISIBLE_DEVICES=1,3
torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 10144 circuit_tuning_llama.py \
    --model_name $MODEL_NAME \
    --model_path "" \
    --device "cuda:$device_id" \
    --task $TASK \
    --enable_fsdp \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --optimizer_name $OPTIMIZER_NAME \
    --accumulation_steps $ACCUMULATION_STEPS \
    --top_n $TOP_N \
    --threshold $THRESHOLD \
    --random_ratio $RAMDOM_RATIO \
    --topn_scheduler $TOPN_START $TOPN_WARMUP $TOPN_TYPE \
    --ablation_method $ABLATION_METHOD \
    --process_or_outcome $PROCESS_OR_OUTCOME \
    --objective $OBJECTIVE \
    --loss_weight_p $LOSS_WEIGHT_P \
    --regularization_beta $REGULARIZATION_BETA \
    --cross_layer \
    --wandb \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --print_every $PRINT_EVERY \
    --save_dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log
