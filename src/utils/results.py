
from math import ceil
from timeit import repeat
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from itertools import product
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Extraction function
def tflog2pandas(file):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(file)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt")
        traceback.print_exc()
    return runlog_data

def all_results(models, datasets):
    df_all = []
    for ds, m in product(datasets, models):
        path = os.path.join("logs/experiments/runs/", ds, m)
        for dirpath, dirnames, filenames in os.walk(path):
            if "run_1" in dirnames:
                df = tflog2pandas(os.path.join(dirpath, "run_1"))
                df["model"] = m
                df["dataset"] = ds
                df_all.append(df)
            
    return pd.concat(df_all)

def extract_metric(df, metric="test/rmse"):
    return df.loc[df.metric==metric]

def format_df(df, format_string="%.3f", mode="min"):
    for col in df:
        data = df[col]
        if mode=="max":
            extrema = data != data.max()
        elif mode=="min":
            extrema = data != data.min()
        else:
            extrema=[True for _ in data]
        bolded = data.apply(lambda x : "\\textbf{%s}" % format_string % x)
        formatted = data.apply(lambda x : format_string % x)
        df[col] = formatted.where(extrema, bolded) 
    return df

def plot_horizon_rmse(df, datasets, metrics, fontsize=15, figname="default"):
    model_dict = {
        "persistence": "Persistence",
        "cnn": "CNN",
        "distana": "Distana",
        "convLSTM": "ConvLSTM",
        "resnet": "ResNet",
        "pdenet": "PDE-Net 2.0",
        "hiddenstate": "Hidden State",
        "neuralPDE": "NeuralPDE"
    }
    color_dict = {
        "persistence": u'#bcbd22',
        "cnn": u'#e377c2',
        "distana": u'#8c564b',
        "convLSTM": u'#9467bd',
        "resnet": u'#d62728',
        "pdenet": u'#2ca02c',
        "hiddenstate": u'#1f77b4',
        "neuralPDE": u'#ff7f0e'
    }
    assert len(datasets) == len(metrics)
    num_plots = len(datasets)
    #with plt.style.context("seaborn-ticks"):
    with plt.style.context("seaborn-paper"):

        fig, axlist = plt.subplots(ceil(num_plots/2), 2, figsize=(12, 8), dpi=200)
        """
        fig = plt.figure(figsize=(12, 8), dpi=200)
        gs = gridspec.GridSpec(2, 4, figure=fig)
        gs.update(wspace=0.5)
        ax1 = plt.subplot(gs[0, :2])
        ax2 = plt.subplot(gs[0, 2:])
        ax3 = plt.subplot(gs[1, 1:3])
        axlist=np.array([ax1, ax2, ax3])
        """
        index = 0
        for dataset, metric in zip(datasets, metrics):
            current_ax = axlist.flatten()[index]
            df_target = df.loc[(df.metric==metric) * (df.dataset==dataset)]
            df_target = df_target.pivot(index='model', columns='step', values='value')
            target_name = metric.split("/")[-1]

            current_ax.set_title(f"{dataset.replace('_', ' ').title()} - {target_name.title().replace('_', ' ')}")
            current_ax.set_xlabel("Horizon")
            current_ax.set_ylabel("RMSE")
            for item in ([current_ax.title, current_ax.xaxis.label, current_ax.yaxis.label] +
                current_ax.get_xticklabels() + current_ax.get_yticklabels()):
                item.set_fontsize(fontsize)
            ylim = 0
            dashed = False
            for model in df_target.index:
                line_style = '--' if dashed else '-'
                line_style = '-' if model == "neuralPDE" else '--'
                current_ax.plot(df_target.columns, df_target.loc[model], line_style, label=model_dict[model], linewidth=4, color=color_dict[model])
                ylim = max(ylim, max(df_target.loc[model]))
                dashed = not dashed
                
            upper_limit = 1.5 if target_name=="wind_speed_lat" else 1
            upper_limit = 2.2 if target_name=="amplitude" else upper_limit
            upper_limit = 2 if target_name=="quantity" else upper_limit
            ylim = min(upper_limit, ylim)
            current_ax.set_ylim(ymin=0, ymax=ylim)
            #current_ax.legend(fontsize=12, loc=2, bbox_to_anchor=(0.05, 0.975))
            index+=1
        fig.tight_layout()
        fig.subplots_adjust(right=0.8, wspace=0.3, hspace=0.3)  # create some space below the plots by increasing the bottom-value
        handles, labels = axlist.flatten()[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=7, bbox_to_anchor=(1, 0.5), ncol=1, fontsize=fontsize)
        #fig.tight_layout()
        plt.savefig(f"{figname}.png")
    return fig



if __name__ == "__main__":
    datasets = ["advection_diffusion","burgers","gas_dynamics","wave","oceanwave","weatherbench", "plasim"]
    models = ["persistence","cnn","distana","convLSTM","resnet","pdenet","hiddenstate","neuralPDE"]
    #datasets = ["advection_diffusion","burgers","gas_dynamics","wave"]
    #models = ["hiddenstate","neuralPDE", "neuralPDE_2order"]
    df = all_results(models, datasets)
    datasets = [
        "gas_dynamics",
        "weatherbench",
        "oceanwave",
        "plasim"
    ]
    metrics = [
        "test/rmse/density",
        "test/rmse/temperature",
        "test/rmse/velocity_x",
        "test/rmse/velocity_y"
    ]
    datasets = [
        "gas_dynamics",
        "gas_dynamics",
        "gas_dynamics",
        "gas_dynamics",
    ]
    metrics = [
        "test/rmse/density",
        "test/rmse/temperature",
        "test/rmse/velocity_x",
        "test/rmse/velocity_y"
    ]
    datasets = [
        "plasim",
        "plasim",
        "plasim",
        "plasim",
    ]
    metrics = [
        "test/rmse/geopotential",
        "test/rmse/temperature",
        "test/rmse/wind_speed_lon",
        "test/rmse/wind_speed_lat"
    ]
    datasets = [
        "oceanwave",
        "oceanwave",
        "oceanwave",
    ]
    metrics = [
        "test/rmse/height",
        "test/rmse/mean_direction",
        "test/rmse/principal_direction",
    ]
    datasets = [
        "wave",
        "advection_diffusion",
        "burgers",
        "burgers",
    ]
    metrics = [
        "test/rmse/amplitude",
        "test/rmse/quantity",
        "test/rmse/velocity_x",
        "test/rmse/velocity_y",
    ]
    plot_horizon_rmse(df, datasets, metrics, figname="fig_toy")


    quit()
    rmse_df=extract_metric(df)
    rmse_df = rmse_df.reset_index()
    rmse_df = rmse_df.pivot(index='model', columns='dataset', values='value')
    rmse_df=rmse_df[datasets].loc[models]
    rmse_df=format_df(rmse_df, mode="min")
    print(rmse_df.to_latex(escape=False))

    