
cd "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning"

MODEL="gpt2-small"
TASK="sv"
DEVICE="cuda:0"

BATCH_SIZE=16
NUM_EPOCHS=3
LR=1e-3
OPTIMIZER_NAME="SGD"
WARMUP_STEPS=100
ACCUMULATION_STEPS=1

BSZ=$(expr $BATCH_SIZE \* $ACCUMULATION_STEPS)

PRUNE_METHOD="top_nodes"
TOP_N=500
THRESHOLD=0
METRIC="logit_diff"  # logit diff
RAMDOM_RATIO=0
RANDN_KEY=0
TOPN_START=0
TOPN_WARMUP=0
BETA_1=0.9
BETA_2=0.999
ABLATION_METHOD="mean"
OBJECTIVE="weighted"
LOSS_WEIGHT_P=1

WANDB_API_KEY=""
WANDB_PROJECT_NAME="circuit-tuning-$TASK-new"
PRINT_EVERY=5
EVAL_STEPS=100

CKPT_DIR="$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-threshold_$THRESHOLD-metric-$METRIC-random_ratio_$RAMDOM_RATIO-randn_$RANDN_KEY-ablation_$ABLATION_METHOD-loss_${OBJECTIVE}_p_$LOSS_WEIGHT_P-prune_method_$PRUNE_METHOD"


SAVE_DIR="/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/${CKPT_DIR}"

mkdir -p $SAVE_DIR

python circuit_tuning_old.py \
    --model $MODEL \
    --bias \
    --task $TASK \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --optimizer_name $OPTIMIZER_NAME \
    --warmup_steps $WARMUP_STEPS \
    --accumulation_steps $ACCUMULATION_STEPS \
    --prune_method $PRUNE_METHOD \
    --top_n $TOP_N \
    --threshold $THRESHOLD \
    --metric $METRIC \
    --random_ratio $RAMDOM_RATIO \
    --topn_scheduler $TOPN_START $TOPN_WARMUP \
    --beta_1 $BETA_1 \
    --beta_2 $BETA_2 \
    --ablation_method $ABLATION_METHOD \
    --objective $OBJECTIVE \
    --loss_weight_p $LOSS_WEIGHT_P \
    --cross_layer \
    --wandb \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --print_every $PRINT_EVERY \
    --eval_steps $EVAL_STEPS \
    --save_dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log
