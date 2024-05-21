from turtle import color
import wandb
import numpy as np
import os
from collections import defaultdict
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from icecream import ic

plt.switch_backend('agg')
plt.rcParams['pdf.fonttype'] = 42

res_dict = {
    "linear_game_H3_A10_N0.5_T100000_N0.5": {
        "BalancedFTRL_0.0016681005372000592",
        "F2TRL_3.593813663804628e-06_True_max(1.5log(x),5)_G0.001",
        "IXOMD_0.0016681005372000592",
        "AdaptiveFTRL_77.42636826811267",
        "BalancedOMD_0.7742636826811277"
    },
    "linear_game_H5_A5_N0.5_T100000_N0.5": {
        "BalancedFTRL_0.0010000000000000009",
        "BalancedOMD_0.7742636826811277",
        "F2TRL_4.641588833612779e-07_True_max(1.5log(x),5)_G0.001",
        "AdaptiveFTRL_1291549.6650148837",
        "IXOMD_0.0016681005372000592"
    }
}

sns.set(
    style="whitegrid",
    rc={
        "grid.linestyle": ":",
        "grid.linewidth": 2.5,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "axes.facecolor": "#F6F6F6",
        "axes.edgecolor": "#333",
        "legend.fontsize": 20,
        "xtick.labelsize": 16,
        "xtick.bottom": True,
        "xtick.color": "#333",
        "ytick.labelsize": 16,
        "ytick.left": True,
        "ytick.color": "#333",
    },
)

sns.set_theme(style="whitegrid")

ENTITY = "your_wandb_account_name"

red = (214/255,39/255,40/255)
blue = (31/255,119/255,180/255)
grey = (139/255, 139/255, 139/255)
purple = (125/255, 46/255, 141/255)
yellow = (237/255, 176/255, 31/255)
green = (133/255, 171/255, 79/255)

COLOR_PLATE = {
    "BalancedFTRL": grey,
    "F2TRL": red,
    "IXOMD": yellow,
    "AdaptiveFTRL": green,
    "BalancedOMD": blue,
}

LEGEND_PLATE = {
    "BalancedFTRL": "BalancedFTRL",
    "F2TRL": "F2TRL",
    "IXOMD": "IXOMD",
    "AdaptiveFTRL": "AdaptiveFTRL",
    "BalancedOMD": "BalancedOMD",
}

plot_order = ["IXOMD", "BalancedOMD", "BalancedFTRL", "AdaptiveFTRL", "F2TRL"]

task_name_dict = {
    "linear_game_H3_A10_N0.5_T100000_N0.5": r"Linear IIEFG ($A=10$, $H=3$)",
    "linear_game_H5_A5_N0.5_T100000_N0.5": r"Linear IIEFG ($A=5$, $H=5$)",
}

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def draw_one_figure(axs, algorithm_dict, y_content="cum_regret", figname="none", task_name=''):
    sm = 0.1
    for algorithm_name in plot_order:
        df_file_path = '{}_{}_{}.csv'.format(task_name, algorithm_name, str(sm))
        if not os.path.exists(df_file_path):
            interval = 50
            print(colored("processing: %s" % algorithm_name, color="cyan"))
            summaries = []
            seed_list = algorithm_dict[algorithm_name]
            idx_list = [i for i in range(0, int(1e5)-100, interval)]
            for run in seed_list:
                ic('p1')
                history = run.history(keys=[y_content], x_axis="_step", samples=int(1e6))
                history["cum_regret"] = smooth(history["cum_regret"], sm)
                ic('p2')
                summaries.append(history.iloc[idx_list])
            summaries = pd.concat(summaries)
            summaries.index = range(len(summaries))
            summaries.to_csv(df_file_path, index=False)
        else:
            print(colored("processing: %s, reading existing file" % algorithm_name, color="cyan"))
            summaries = pd.read_csv(df_file_path)
            summaries.index = range(len(summaries))
        used_color_plate = COLOR_PLATE
        ic(summaries)
        sns.lineplot(
            data=summaries,
            x="_step",
            y=y_content,
            errorbar=('ci', 95),
            label=LEGEND_PLATE[algorithm_name],
            alpha=0.8,
            color=used_color_plate[algorithm_name],
            ax=axs,
        )
        ic('finish one plot')
    axs.set_yscale('log')


def get_algorithm_name(run):
    return str(run.job_type).split('_')[0]


def main(to_draw_dict, figname, y_limit_dict):
    api = wandb.Api(timeout=60)
    task_dict = defaultdict()
    for env_name in to_draw_dict.keys():
        task_dict[env_name] = "None"
        runs = api.runs(os.path.join(ENTITY, 'IIEFG'))
        algorithm_dict = defaultdict(lambda: "None")
        for run in runs:
            if run.group != env_name:
                continue
            algorithm_name = get_algorithm_name(run)
            if algorithm_dict[algorithm_name] == "None":
                algorithm_dict[algorithm_name] = []
            algorithm_dict[algorithm_name].append(run)
        copy_dict = defaultdict()
        for key in sorted(algorithm_dict.keys()):
            copy_dict[key] = algorithm_dict[key]
        task_dict[env_name] = copy_dict
    figure_length = len(task_dict.keys())
    figure, axes = plt.subplots(1, figure_length, figsize=(15, 5))
    idx = 0
    for task_name in task_dict.keys():
        algorithm_dict = task_dict[task_name]
        axs = axes[idx]
        axs.set_title(task_name_dict[task_name])
        draw_one_figure(
            axs,
            algorithm_dict,
            y_content="cum_regret",
            figname=figname,
            task_name=task_name
        )
        axs.set_ylabel("Cumulative Regret")
        axs.set_xlabel("Episodes")
        axs.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
        axs.set_ylim(y_limit_dict[task_name])
        idx += 1
    plt.tight_layout()
    plt.savefig("imgs/%s.png" % figname)
    plt.savefig("pdfs/%s.pdf" % figname)
    plt.yscale('log')
    plt.close()
    print(colored("figure saved in %s." % figname, color="red"))


if __name__ == "__main__":
    pairs = [
        (
            "linear_iiefg",
            res_dict,
            { 
                "linear_game_H3_A10_N0.5_T100000_N0.5": (1, 50000),
                "linear_game_H5_A5_N0.5_T100000_N0.5": (1, 50000),
            },
        ),
    ]
    for figname, to_draw_dict, y_limit_dict in pairs:
        main(to_draw_dict, figname, y_limit_dict)
