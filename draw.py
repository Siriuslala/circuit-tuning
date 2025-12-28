"""
Draw figures.
"""

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline
import numpy as np
import jsonlines

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path='./.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(str(root_dir))

# http://zxqsq.wiicha.com/

def draw_topn_mem(eap_new=False):

    topn_list = [10, 15, 20, 30, 50, 70, 100, 500, 1000, 5000, 30000]
    if eap_new:
        topn_list = [10, 15, 20, 50, 100, 200, 300, 400, 500, 1000, 2500, 5000, 30000]
    else:
        topn_list = [10, 15, 20, 50, 100, 500, 1000, 5000, 30000]

    ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"

    average_params_ratio_list = []
    nodes_num_list = []
    logit_diff_list = []
    ppl_list = []
    for topn in topn_list:
        log_path = ckpt_path.format(topn=topn)
        print(topn)
        with jsonlines.open(log_path) as f:
            data = [line for line in f]
        average_params_ratio_list.append(data[-1]["avg_ideal_tuning_params_num"] / data[-1]["model_params_num"])
        # nodes_num_list.append(data[-1]["nodes_num"])
        diff = 0
        for line in data:
            if "eval_result" in line:
                diff = line["eval_result"]["logit_diff"]
                ppl = line["eval_result"]["ppl"]   
        logit_diff_list.append(diff)
        ppl_list.append(ppl)
    print(average_params_ratio_list)
        
    # draw
    # plt.rcParams['font.style'] = 'italic'

    # param ratio
    fig, ax1 = plt.subplots()
    color = '#457855'
    ax1.set_xlabel('Top n', loc="right", color=mcolors.to_rgba('black', 0.7))
    # ax1.set_ylabel('Average parameters ratio', color=color)
    ax1.set_xlim(right=10000)
    ax1.set_ylim(top=0.59) 
    line1, = ax1.plot(topn_list, average_params_ratio_list, color=color, label='Average parameters ratio')

    y_line = 0.457
    ax1.axhline(y=y_line, color=color, linestyle='--')
    ax1.text(0, y_line, f' {y_line}', color=color, verticalalignment='bottom')

    ax1.tick_params(axis='y', labelcolor=color)


    # logit diff
    ax2 = ax1.twinx()  
    color = 'tab:green'  # '#CBB1A5'
    # spl = make_interp_spline(topn_list, logit_diff_list, k=3)  # k是插值多项式的阶数
    # x_smooth = np.linspace(0, 5000, 20)  # 在新的X轴范围内生成更密集的点
    # y_smooth = spl(x_smooth)  # 使用插值对象生成平滑的Y值
    # ax2.plot(x_smooth, y_smooth, color=color)
    # ax2.set_ylabel('Final logit difference', color=color)
    ax2.set_ylim(top=3, bottom=-0.3)
    line2, = ax2.plot(topn_list, logit_diff_list, color=color, label='Final logit difference')
    ax2.tick_params(axis='y', labelcolor=color)


    # ppl
    ax3 = ax1.twinx()
    color = '#926A53'
    ax3.spines['right'].set_position(('axes', 1.1))
    # ax3.set_ylabel('PPL', color=color)
    ax3.set_ylim(top=80.01, bottom=71.5)
    line3, = ax3.plot(topn_list, ppl_list, color=color, label='PPL')
    ax3.tick_params(axis='y', labelcolor=color)
    
    ax1.spines['top'].set_color((0.3, 0.3, 0.3))
    ax2.spines['top'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    
    ax1.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax2.spines['bottom'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    
    ax1.spines["left"].set_color((0.3, 0.3, 0.3))
    ax2.spines['left'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    ax1.spines["right"].set_color((0.3, 0.3, 0.3))
    ax2.spines['right'].set_visible(False)
    
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    ax1.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax1.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
        
    # plt.title('Top n influence on average parameters ratio and final logit difference')
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    legend = plt.legend(lines, labels, loc='upper right')
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))

    plt.savefig("./figures/topn_influence_new.pdf", bbox_inches='tight')

def draw_topn(var="logit_diff"):
    
    all_data = []

    ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    ckpt_path = f"{work_dir}/checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    topn_list = [10, 15, 20, 50, 100, 500, 1000, 2500, 10000]
    for topn in topn_list:
        log_path = ckpt_path.format(topn=topn)
        print(topn)
        with jsonlines.open(log_path) as f:
            data = [line for line in f if "eval_result" in line]
            all_data.append(data)
    steps = [i*100 for i in range(1, len(all_data[0])+1)]
    # points = [len(data) for data in all_data]
    # print(points)

    # full, lr=5e-4
    full_tuning_path = f"{work_dir}/checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_5e-4-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning/log.jsonl"
    with jsonlines.open(full_tuning_path) as f:
        data = [line for line in f if "eval_result" in line]
        all_data.append(data)

    # lr=1e-3
    full_tuning_path = f"{work_dir}/checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning/log.jsonl"
    with jsonlines.open(full_tuning_path) as f:
        data = [line for line in f if "eval_result" in line]
        all_data.append(data)
    
    
    all_data = [[line["eval_result"][var]  for line in data] for data in all_data]
    
    color_list = ["#479A5F", "#A1A9AD", "#EDB732", "#050505", "#5BC5DB", "#87CEBF", "#A0C75C", "#E87B9F"]  # wandb
    color_list = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeed", "#f2a7da"]# too bright
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#eac47c", "#d7d3b0", "#abc5aa"]
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#d7d3b0", "#f9d580", "#abc5aa"]
    color_list = ["#A1A9AD", "#613B25", "#E98956", "#B47440", "#CCAA8A", "#68BE23", "#FABF02", "#9BC750", "#56A318", "#9CD5F1", "#6EB0D1"]
    # color_list = ["#A1A9AD", "#613B25", "#E98956", "#B47440", "#CCAA8A", "#68BE23", "#FABF02", "#9BC750", "#56A318", "#356F05", "#6EB0D1"]

    # draw
    # plt.rcParams['font.style'] = 'italic'

    for i, data in enumerate(all_data):
        if i == len(all_data) - 1:
            label = "Full fine-tuning (lr=1e-3)"
        elif i == len(all_data) - 2:
            label = "Full fine-tuning (lr=5e-4)"
        else:
            label = f"Top-N={topn_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data[i], color=color, marker='o', markevery=slice(-1, None, None), label=label)

    ticks = np.arange(0, max(steps)+1, 1000)
    x_labels = [f'{tick/1000:.0f}k' for tick in ticks]
    plt.xticks(ticks, x_labels)
    
    plt.xlim(left=101)
    # plt.ylim(bottom=-3.9999)
    
    plt.xlabel("Steps", loc="right", color=mcolors.to_rgba('black', 0.7))
    # plt.ylabel("Logit difference", loc="top")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))

    save_dir = "./figures/basic/fL0b"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/topn_{var}.pdf", bbox_inches='tight')

def draw_num_nodes(var="nodes_num", eap_new=False):
    
    all_data = []
    
    if eap_new:
        ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes_new1/log.jsonl"
        topn_list = [15, 20, 100, 500, 1000, 2500, 5000]
    else:
        ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
        ckpt_path = f"{work_dir}/checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
        topn_list = [10, 15, 20, 50, 100, 500, 1000, 2500, 5000, 10000]
    for topn in topn_list:
        log_path = ckpt_path.format(topn=topn)
        print(topn)
        with jsonlines.open(log_path) as f:
            data = [line for line in f]
            all_data.append(data)
    all_data = [[line[var] for line in data if var in line]  for data in all_data]
    points = [len(data) for data in all_data]
    print(points)
    min_points = min(points)
    all_data = [data[-min_points:] for data in all_data]

    steps = [i for i in range(1, len(all_data[0])+1)]
    
    color_list = ["#479A5F", "#A1A9AD", "#EDB732", "#050505", "#5BC5DB", "#87CEBF", "#A0C75C", "#E87B9F"]  # wandb
    color_list = ["#353130", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeed", "#f2a7da"]# too bright
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#eac47c", "#d7d3b0", "#abc5aa"]
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#d7d3b0", "#f9d580", "#abc5aa"]
    color_list = ["#A1A9AD", "#613B25", "#E98956", "#B47440", "#CCAA8A", "#68BE23", "#FABF02", "#9BC750", "#56A318", "#356F05"]
    
    # draw
    # plt.rcParams['font.style'] = 'italic'

    for i, data in enumerate(all_data):
        label = f"Top-N={topn_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data[i], color=color, marker='o', markevery=slice(-1, None, None), label=label)

    ticks = np.arange(0, max(steps)+1, 1000)
    x_labels = [f'{tick/1000:.0f}k' for tick in ticks]
    plt.xticks(ticks, x_labels)
    
    plt.xlim(left=101)
    # plt.ylim(bottom=-3.9999)
    
    plt.xlabel("Steps", loc="right", color=mcolors.to_rgba('black', 0.7))
    # plt.ylabel("Logit difference", loc="top")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))

    save_dir = "./figures/basic/fL0b"
    save_path = f"{save_dir}/topn_{var}.pdf" if not eap_new else f"{save_dir}/topn_{var}_new.pdf"
    plt.savefig(save_path, bbox_inches='tight')

def draw_randn(var="ppl", draw_std=True):
    """
    topn1000, randn=0.1, 0.2, 0.3, 0.4
    """
    all_data = []

    full_tuning_path = f"{work_dir}/checkpoints-sv/gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-loss_weighted(p_1)-full_tuning/log.jsonl"
    with jsonlines.open(full_tuning_path) as f:
        data = [line for line in f if "eval_result" in line]
        all_data.append(data)

    origin_tuning_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_8-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    with jsonlines.open(origin_tuning_path) as f:
        data = [line for line in f if "eval_result" in line]
        all_data.append(data)

    ckpt_dir = str(work_dir / "checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-k_8-threshold_0-metric-logit_diff-random_ratio_{random_ratio}-randn_{randn_idx}-ablation_mean-loss_weighted_p_1-prune_method_top_edges")

    r_list = [0.05, 0.1, 0.2, 0.3, 0.4]
    # key_list = [0, 1, 2, 3, 4]
    key_list = [0, 1, 2, 3]

    randn_data = [[] for _ in range(len(r_list))]
    print(randn_data)
    for i, r in enumerate(r_list):
        for key in key_list:
            log_path = ckpt_dir.format(random_ratio=r, randn_idx=key) + "/log.jsonl"
            print(log_path)
            if not os.path.exists(log_path):
                continue
            with jsonlines.open(log_path) as f:
                data = [line for line in f if "eval_result" in line]
                randn_data[i].append(data)

    steps = [i*100 for i in range(1, len(all_data[0])+1)]
    all_data = [[line["eval_result"][var] for line in data] for data in all_data]
    # For each random ratio, average over multiple random seeds
    for i, data_group in enumerate(randn_data):  # group: [[{"eval": {"ppl": ...}}, {"eval": {"ppl": ...}}], [{"eval": {"ppl": ...}}]]
        # Each data_group contains multiple runs with different random seeds
        data_num = max(len(data) for data in data_group)
        # drop the data that doesn't have enough points
        data_group = [data for data in data_group if len(data) == data_num]
        data_group_var = [[line["eval_result"][var] for line in data] for data in data_group]
        data_group_var_mean = np.mean(data_group_var, axis=0)
        data_group_var_std = np.std(data_group_var, axis=0, ddof=1)
        # data_group_var_std_error = data_group_var_std / np.sqrt(data_num)
        randn_data[i] = [data_group_var_mean, data_group_var_std]
        print(i)
    all_data.extend(randn_data)
    
    color_list = ["#479A5F", "#050505", "#E87B9F", "#EDB732", "#E57439", "#5BC5DB", "#5BDB64"]
    
    
    # draw
    # plt.rcParams['font.style'] = 'italic'

    for i, data in enumerate(all_data):
        if i == 0:
            continue
            label = "full-tuning"
        elif i == 1:
            label = "random ratio=0"
        else:
            label = f"random ratio={r_list[i-2]}"
        color = color_list[i]
        # linewitdth = 3 if i == 1 else 1.5
        linewitdth = 1.5
        order = 6 if i == 1 else None
        if i <= 1:
            plt.plot(steps, all_data[i], linewidth=linewitdth, color=color, marker='o', markevery=slice(-1, None, None), label=label, zorder=order)
        else:
            plt.plot(steps, all_data[i][0], linewidth=linewitdth, color=color, marker='o', markevery=slice(-1, None, None), label=label, zorder=order)
            if draw_std:
                plt.fill_between(steps, all_data[i][0] - all_data[i][1], all_data[i][0] + all_data[i][1], color=color, alpha=0.2)

    ticks = np.arange(0, max(steps)+1, 1000)
    x_labels = [f'{tick/1000:.0f}k' for tick in ticks]
    plt.xticks(ticks, x_labels)
    
    plt.xlim(left=101)
    # plt.ylim(bottom=-3.9999)
    
    plt.xlabel("Steps", loc="right", color=mcolors.to_rgba('black', 0.7))
    # plt.ylabel("Logit difference", loc="top")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))

    save_dir = "./figures/random_activate"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/random_ratio_{var}.pdf" if not draw_std else f"{save_dir}/random_ratio_{var}_draw_std.pdf"
    plt.savefig(save_path, bbox_inches='tight')
        
def draw_topn_new():
    """
    draw EAP new (old)
    """
    
    all_data_old = []
    all_data_new = []
    
    log_path_old = [
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_50-threshold_0-random_ratio_0-ablation_{{mean}}-loss_weighted(p_1)/log.jsonl",
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_100-threshold_0-random_ratio_0-ablation_{{mean}}-loss_weighted(p_1)/log.jsonl",
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_500-threshold_0-random_ratio_0-ablation_{{mean}}-loss_weighted(p_1)/log.jsonl",
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_1000-threshold_0-random_ratio_0-ablation_{{mean}}-loss_weighted(p_1)/log.jsonl"
        ]
    log_path_new = [
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_50-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/log.jsonl",
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_100-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/log.jsonl",
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_500-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/log.jsonl",
        f"{work_dir}/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_{{SGD}}-warm_up_100-top_n_1000-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/log.jsonl",
    ]
    topn_list = [100, 500, 1000]

    for log_path in log_path_old:
        print(log_path)
        with jsonlines.open(log_path) as f:
            data = [line for line in f if "eval_result" in line]
            all_data_old.append(data)
    for log_path in log_path_new:
        print(log_path)
        with jsonlines.open(log_path) as f:
            data = [line for line in f if "eval_result" in line]
            all_data_new.append(data)
            
    steps = [i*100 for i in range(1, len(all_data_old[0])+1)]
    all_data_old = [[line["eval_result"]["logit_diff"]  for line in data][-len(steps):]  for data in all_data_old][1:]
    all_data_new = [[line["eval_result"]["logit_diff"]  for line in data][-len(steps):]  for data in all_data_new][1:]
    print(len(all_data_old[0]))
    print(len(all_data_new[0]))
    
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#d7d3b0", "#f9d580", "#abc5aa"]

    color_list = ["#299d8f", "#e9c46a", "#d87659"]
    color_list = ["#14bc94", "#e2d5ba", "#f6a9cd"]
    color_list = ["#678983", "#e6ddc4", "#93ae9f"]
    color_list = ["#4f3a29", "#b1a369", "#abc5aa"]
    
    # draw
    for i, data in enumerate(all_data_old):
        label = f"topn={topn_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data_old[i], color=color, linestyle='--', marker='o', markevery=slice(-1, None, None), label=label)
    for i, data in enumerate(all_data_new):
        label = f"new-topn={topn_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data_new[i], color=color, linestyle='-', marker='o', markevery=slice(-1, None, None), label=label)

    ticks = np.arange(0, max(steps)+1, 1000)
    x_labels = [f'{tick/1000:.0f}k' for tick in ticks]
    plt.xticks(ticks, x_labels)
    
    plt.xlim(left=101)
    # plt.ylim(bottom=-3.9999)
    
    plt.xlabel("Steps", loc="right", color=mcolors.to_rgba('black', 0.7))
    # plt.ylabel("Logit difference", loc="top")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))

    plt.savefig("Interpretability/lyy/circuit-tuning/figures/topn_new.pdf", bbox_inches='tight')

def draw_EAP_new(var, ig=False):
    
    all_data_old = []
    all_data_new = []
    
    log_path_old = str(work_dir) + "/checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{topn}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    if ig:  # EAP-IG
        log_path_new = str(work_dir) + "/checkpoints-sv/ig-fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{topn}-k_16-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    else:
        log_path_new = str(work_dir) + "/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{topn}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes_new1/log.jsonl"
    # topn_list = [15, 20, 100, 500, 1000, 2500, 5000]
    # topn_list = [15, 20, 100, 500, 1000, 2500]
    topn_list = [15, 20, 100, 500, 1000]
    topn_list = [50, 100, 500, 1000]

    for topn in topn_list:
        log_path = log_path_old.format(topn=topn)
        print(topn)
        with jsonlines.open(log_path) as f:
            data = [line for line in f if "eval_result" in line]
            all_data_old.append(data)
    for topn in topn_list:
        log_path = log_path_new.format(topn=topn)
        print(topn)
        with jsonlines.open(log_path) as f:
            data = [line for line in f if "eval_result" in line]
            all_data_new.append(data)
            
    all_data_old = [[line["eval_result"][var] for line in data] for data in all_data_old]
    all_data_new = [[line["eval_result"][var] for line in data] for data in all_data_new]
    steps = [i*100 for i in range(1, len(all_data_old[0])+1)]
    
    color_list = ["#4f3a29", "#b1a369", "#abc5aa"]
    color_list = ["#A1A9AD", "#613B25", "#E98956", "#B47440", "#CCAA8A", "#68BE23", "#FABF02", "#9BC750", "#56A318", "#6EB0D1"]
    color_list = ["#4f3a29", "#b1a369", "#A1A9AD", "#abc5aa"]
    
    # draw
    for i, data in enumerate(all_data_old):
        label = f"topn={topn_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data_old[i], color=color, linestyle='--', marker='o', markevery=slice(-1, None, None), label=label)
    for i, data in enumerate(all_data_new):
        label = f"new-topn={topn_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data_new[i], color=color, linestyle='-', marker='o', markevery=slice(-1, None, None), label=label)

    ticks = np.arange(0, max(steps)+1, 1000)
    x_labels = [f'{tick/1000:.0f}k' for tick in ticks]
    plt.xticks(ticks, x_labels)
    
    plt.xlim(left=101)
    # plt.ylim(bottom=-3.9999)
    
    plt.xlabel("Steps", loc="right", color=mcolors.to_rgba('black', 0.7))
    # plt.ylabel("Logit difference", loc="top")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))

    save_dir = "./figures/EAP-IG" if ig else "./figures/EAP-new"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/topn_{var}.pdf", bbox_inches='tight')

def draw_k(var="logit_diff", topn=500, bsz=16):
    
    all_data = []

    ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_{bsz}-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_{{k}}-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    k_list = [1, 4, 8, 16, 32, 64, 128]
    for k in k_list:
        log_path = ckpt_path.format(topn=topn, k=k)
        print(topn, k)
        with jsonlines.open(log_path) as f:
            data = [line for line in f if "eval_result" in line]
            all_data.append(data)
    steps = [i*100 for i in range(1, len(all_data[0])+1)]
    
    all_data = [[line["eval_result"][var]  for line in data]  for data in all_data]
    
    color_list = ["#479A5F", "#A1A9AD", "#EDB732", "#050505", "#5BC5DB", "#87CEBF", "#A0C75C", "#E87B9F"]  # wandb
    color_list = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeed", "#f2a7da"]# too bright
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#eac47c", "#d7d3b0", "#abc5aa"]
    color_list = ["#479A5F", "#797657", "#4f3a29", "#b69272", "#b1a369", "#d7d3b0", "#f9d580", "#abc5aa"]
    color_list = ["#A1A9AD", "#613B25", "#E98956", "#B47440", "#CCAA8A", "#68BE23", "#FABF02", "#9BC750", "#56A318", "#6EB0D1"]
    
    # draw
    # plt.rcParams['font.style'] = 'italic'

    for i, data in enumerate(all_data):
        label = f"Top-N={topn}, K={k_list[i]}"
        color = color_list[i]
        plt.plot(steps, all_data[i], color=color, marker='o', markevery=slice(-1, None, None), label=label)

    ticks = np.arange(0, max(steps)+1, 1000)
    x_labels = [f'{tick/1000:.0f}k' for tick in ticks]
    plt.xticks(ticks, x_labels)
    
    plt.xlim(left=101)
    # plt.ylim(bottom=-3.9999)
    
    plt.xlabel("Steps", loc="right", color=mcolors.to_rgba('black', 0.7))
    # plt.ylabel("Logit difference", loc="top")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))
    save_dir = "./figures/K"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/topn_{topn}_bsz_{bsz}_{var}.pdf", bbox_inches='tight')

def draw_intrinsic(eap_new=False):
    saturation_limit = 0.455
    if eap_new:
        ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes_new1/log.jsonl"
        topn_list = [20, 100, 200, 300, 400, 500, 1000, 2500, 5000]
    else:
        topn_list = [10, 15, 20, 50, 100, 500, 1000, 2500, 5000, 10000]
        ckpt_path = f"{work_dir}/checkpoints-sv/QKVO-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
        ckpt_path = f"{work_dir}/checkpoints-sv/fL0b-gpt2-small-sv-epochs_6-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_{{topn}}-k_1-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_edges/log.jsonl"
    logit_diff_list = []
    ppl_list = []
    nodes_num_list = []
    ideal_param_ratio_list = []
    for topn in topn_list:
        log_path = ckpt_path.format(topn=topn, k=1)
        print(topn)
        with jsonlines.open(log_path) as f:
            data = list(f)
            eval_data = [line for line in data if "eval_result" in line]
        logit_diff_list.append(eval_data[-1]["eval_result"]["logit_diff"])
        ppl_list.append(eval_data[-1]["eval_result"]["ppl"])
        nodes_num_list.append(data[-1]["avg_nodes_num"])
        ideal_param_ratio_list.append(data[-1]["avg_ideal_tuning_params_num"] / data[-1]["model_params_num"])
    print(logit_diff_list)
    print(ppl_list)
    print(nodes_num_list)
    print(ideal_param_ratio_list)
    
    data_list = [logit_diff_list, ppl_list, nodes_num_list, ideal_param_ratio_list]
    var_names = ["logit_diff", "ppl", "nodes_num", "ideal_param_ratio"]
    var_labels = ["Logit difference", "Perplexity", "Number of nodes", "Parameter ratio"]
    for var, var_data, label in zip(var_names, data_list, var_labels):
        plt.figure(figsize=(10, 7))

        plt.plot(
            topn_list,
            var_data,
            color='blue',                   # 蓝线
            linestyle='-',                  # 实线
            marker='o',                     # 圆形标记
            markersize=10,                  # 标记大小
            linewidth=2,
            markeredgecolor='darkblue',     # 标记边缘颜色
            markerfacecolor='lightblue',    # 标记填充色
            alpha=0.7,                      # 标记透明度
            label=label
        )

        # plt.axhline(
        #     y=saturation_limit,
        #     color='black',
        #     linestyle='--',
        #     linewidth=1.5
        # )

        plt.xlabel("TopN", fontsize=14, loc="right")
        plt.ylabel(label, fontsize=14)

        plt.grid(
            True,
            linestyle='-',
            alpha=0.6,
            color='#cccccc'
        )

        plt.xticks(np.arange(0, 10001, 5000))

        # plt.ylim(0.08, 0.47)
        # plt.yticks(np.arange(0.15, 0.50, 0.05)) # 设置主刻度

        plt.legend(fontsize=12, loc='lower right')

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.tight_layout()
        save_dir = "./figures/basic/fL0b"
        save_path = f"{save_dir}/intrinsic_{var}.pdf"
        if eap_new:
            save_path = f"{save_dir}/intrinsic_{var}_new.pdf"
        plt.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
    pass

    # draw_randn(var="logit_diff")
    # draw_topn_mem()
    # draw_topn_new()

    # ======================================================
    # draw_topn("ppl")
    # draw_num_nodes("ideal_param_ratio", eap_new=False)  # "ideal_param_ratio", "nodes_num"
    # draw_intrinsic(eap_new=False)

    # draw_randn(var="logit_diff", draw_std=False)
    # draw_randn(var="logit_diff", draw_std=True)

    # draw_EAP_new("logit_diff", ig=True)
    draw_k(topn=5000, bsz=32)
    