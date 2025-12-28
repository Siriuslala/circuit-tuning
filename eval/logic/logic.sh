


python logic.py \
    --model_name "llama-3.1-8b-it" \
    --ckpt_path "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-logic/llama3.1-8b-it-logic-epochs_1-bsz_32-lr_3e-06-Opt_SGD-full-precision_bf16-1/fsdp-/raid_sdd/lyy/hf/models--meta-llama--Llama-3.1-8B-instruct" \
    --device 'cuda:0' \
    --output_dir "llama-3.1-8b-it-full" \

    # peft
    # /raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-logic/llama-3.2-1B-it-logic-epochs_1-bsz_16-lr_0.0001-Opt_SGD-lora_r16_alpha16-precision_bf16-max_train_step_1000/peft_1

    # circuit
    # /raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-logic/Circuit-Llama-3.2-1B-instruct-logic-epochs_3-bsz_16-lr_3e-5-Opt_SGD-top_n_4000-topn_start_0-warmup_0-cosine-guide_process-ablation_mean-ie_over_seq-loss_weighted_p_1-prune_top_nodes/fsdp-epoch_1000steps