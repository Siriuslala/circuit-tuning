
cd "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning"

MODEL="gpt2-small"
TASK="math"
DATA_DIR="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/math_dataset"
DEVICE="gpu"
GPU_IDS="1"

BATCH_SIZE=16
NUM_EPOCHS=10
LR=1e-5
OPTIMIZER_NAME="SGD"
WARMUP_STEPS=100
ACCUMULATION_STEPS=1

BSZ=$(expr $BATCH_SIZE \* $ACCUMULATION_STEPS)

OBJECTIVE="weighted"
LOSS_WEIGHT_P=1

WANDB_API_KEY="f33d04e8bfbd4c33be09b9d9f41445e390346cf6"
WANDB_PROJECT_NAME="circuit-tuning-$TASK"
PRINT_EVERY=5
EVAL_STEPS=100

# no ie_over_seq
CKPT_DIR="$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_{$OPTIMIZER_NAME}-warm_up_$WARMUP_STEPS-loss_$OBJECTIVE(p_$LOSS_WEIGHT_P)-full_tuning"


SAVE_DIR="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-math/${CKPT_DIR}"

mkdir -p $SAVE_DIR

# need to change: lr, batch_size
python full_tuning.py \
    --model $MODEL \
    --task $TASK \
    --data_dir $DATA_DIR \
    --device $DEVICE \
    --gpu_ids $GPU_IDS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --optimizer_name $OPTIMIZER_NAME \
    --warmup_steps $WARMUP_STEPS \
    --accumulation_steps $ACCUMULATION_STEPS \
    --objective $OBJECTIVE \
    --loss_weight_p $LOSS_WEIGHT_P \
    --wandb \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --print_every $PRINT_EVERY \
    --eval_steps $EVAL_STEPS \
    --save_dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log
