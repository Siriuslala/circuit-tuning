source exp_info.sh

cd ..

export WANDB_BASE_URL="https://api.bandw.top"

MODEL="gpt2-small"
TASK="sv"
DEVICE="cuda:3"

BATCH_SIZE=16
NUM_EPOCHS=6
LR=1e-3
OPTIMIZER_NAME="SGD"
WARMUP_STEPS=100
ACCUMULATION_STEPS=1

BSZ=$(expr $BATCH_SIZE \* $ACCUMULATION_STEPS)

PRUNE_METHOD="top_edges"  # top nodes -> EAP new
TOP_N=5000
K=8
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

WANDB_PROJECT_NAME="circuit-tuning-$TASK-new-2025"
PRINT_EVERY=5
EVAL_STEPS=100

# smooth
CKPT_DIR="$MODEL$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-threshold_$THRESHOLD-random_ratio_$RAMDOM_RATIO-beta_1_$BETA_1-beta_2_$BETA_2"

# ie_over_seq no ablation
CKPT_DIR="$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-threshold_$THRESHOLD-random_ratio_$RAMDOM_RATIO-ie_over_seq-ablation_None-loss_$OBJECTIVE(p_$LOSS_WEIGHT_P)"

# ie_over_seq
CKPT_DIR="$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-threshold_$THRESHOLD-random_ratio_$RAMDOM_RATIO-ie_over_seq-ablation_$ABLATION_METHOD-loss_$OBJECTIVE(p_$LOSS_WEIGHT_P)"

# no ie_over_seq, no topn
CKPT_DIR="$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-threshold_$THRESHOLD-random_ratio_$RAMDOM_RATIO-randn_$RANDN_KEY-ablation_$ABLATION_METHOD-loss_${OBJECTIVE}_p_$LOSS_WEIGHT_P"

# no ie_over_seq, no topn, 
CKPT_DIR="$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-k_$K-threshold_$THRESHOLD-metric-$METRIC-random_ratio_$RAMDOM_RATIO-randn_$RANDN_KEY-ablation_$ABLATION_METHOD-loss_${OBJECTIVE}_p_$LOSS_WEIGHT_P-prune_method_$PRUNE_METHOD"

# split Q/K/V
CKPT_DIR="QKVO_no_bias-$MODEL-$TASK-epochs_$NUM_EPOCHS-bsz_$BSZ-lr_$LR-Opt_$OPTIMIZER_NAME-warm_up_$WARMUP_STEPS-top_n_$TOP_N-k_$K-threshold_$THRESHOLD-metric-$METRIC-random_ratio_$RAMDOM_RATIO-randn_$RANDN_KEY-ablation_$ABLATION_METHOD-loss_${OBJECTIVE}_p_$LOSS_WEIGHT_P-prune_method_$PRUNE_METHOD"


SAVE_DIR="$WORK_DIR/checkpoints-sv/${CKPT_DIR}"

mkdir -p $SAVE_DIR

# wait to be added: smooth, ie_over_seq, 
# need to change: lr, batch_size, top_n, threshold, random_ratio, beta_1, beta_2

# wandb offline
# --cross_layer \
python circuit_tuning_old.py \
    --model $MODEL \
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
    --prune_every_k $K \
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
