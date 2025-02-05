
cd "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning"

MODEL_NAME="meta-llama/Llama-3.2-1B-instruct" # "meta-llama/Llama-3.1-8B-instruct"  # "meta-llama/Meta-Llama-3-8B"
MODEL_PATH="/raid_sdd/lyy/hf/models--meta-llama--Llama-3.2-1B-instruct"
TASK="math"

NUM_EPOCHS=3
LR=1e-5
BATCH_SIZE=1
ACCUMULATION_STEPS=8
N_DEVICES=4

OPTIMIZER_NAME="SGD"

BSZ=$(expr $BATCH_SIZE \* $ACCUMULATION_STEPS \* $N_DEVICES)

PRUNE_METHOD="top_nodes"
TOP_N=4000 #
THRESHOLD=0
RAMDOM_RATIO=0
RANDN_KEY=0
TOPN_START=0 #
TOPN_WARMUP=0  # 
TOPN_TYPE="cosine"
PROCESS_OR_OUTCOME="process"
ABLATION_METHOD="mean"
OBJECTIVE="weighted"
LOSS_WEIGHT_P=1

WANDB_API_KEY=""
WANDB_PROJECT_NAME="circuit-tuning-$TASK"
PRINT_EVERY=10

# ckpt_name
IFS='/' read -ra PARTS <<< "$MODEL_NAME"
MODEL=${PARTS[-1]}

CKPT_NAME="Circuit-$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-top_n_$TOP_N-topn_start_$TOPN_START-warmup_$TOPN_WARMUP-$TOPN_TYPE-guide_$PROCESS_OR_OUTCOME-ablation_$ABLATION_METHOD-ie_over_seq-loss_${OBJECTIVE}_p_$LOSS_WEIGHT_P-prune_$PRUNE_METHOD"

SAVE_DIR="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/${CKPT_NAME}"

mkdir -p $SAVE_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes 1 --nproc_per_node $N_DEVICES --master_port 44441 circuit_tuning_llama_new.py \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --ckpt_path "" \
    --task $TASK \
    --enable_fsdp \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --optimizer_name $OPTIMIZER_NAME \
    --accumulation_steps $ACCUMULATION_STEPS \
    --prune_method $PRUNE_METHOD \
    --top_n $TOP_N \
    --threshold $THRESHOLD \
    --random_ratio $RAMDOM_RATIO \
    --topn_scheduler $TOPN_START $TOPN_WARMUP $TOPN_TYPE \
    --ablation_method $ABLATION_METHOD \
    --ie_over_seq \
    --process_or_outcome $PROCESS_OR_OUTCOME \
    --objective $OBJECTIVE \
    --loss_weight_p $LOSS_WEIGHT_P \
    --cross_layer \
    --wandb \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --print_every $PRINT_EVERY \
    --save_dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log
