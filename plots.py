# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
scenario_names = [
    "large files 1 large patterns",
    "large files 100 small patterns",
    "small files 10 large patterns",
    "small files 1000 small patterns",
]
results_gpu_percentage = pd.read_csv("results_gpu_percentage_v4.csv")
for name in scenario_names:
    y = results_gpu_percentage[results_gpu_percentage["scenario"] == name]["mean_time"]
    yerr = results_gpu_percentage[results_gpu_percentage["scenario"] == name]["std_time"]
    x = results_gpu_percentage[results_gpu_percentage["scenario"] == name]["percentage_gpu"]
    plt.errorbar(x, y, yerr=yerr, label=name)
plt.legend()
plt.show()
# %%
# with open("results_thread_block_v4.csv", "w") as f:
#     f.write("scenario,thread_per_block,block_per_grid,mean_time,std_time\n")
results_thread_block = pd.read_csv("results_thread_block_v4.csv")
thread_per_block_values = results_thread_block["thread_per_block"].unique()
block_per_grid_values = results_thread_block["block_per_grid"].unique()
for name in scenario_names:
    sub_df = results_thread_block[results_thread_block["scenario"] == name]
    heatmap = np.zeros((len(thread_per_block_values), len(block_per_grid_values)))
    for i, row in sub_df.iterrows():
        heatmap[
            thread_per_block_values == row["thread_per_block"],
            block_per_grid_values == row["block_per_grid"],
        ] = row["mean_time"]
    plt.imshow(heatmap, vmin=0.37, vmax=0.5)
    plt.colorbar()
    plt.xticks(range(len(block_per_grid_values)), block_per_grid_values)
    plt.xlabel("block_per_grid")
    plt.yticks(range(len(thread_per_block_values)), thread_per_block_values)
    plt.ylabel("thread_per_block")
    plt.title(name)
    plt.show()
# %%
### Perf vs ndata & npattern when distributed vs not, at constant workload
results_dist_pattern = pd.read_csv("results_distributed_and_pattern_v4.csv")
distribute_labels = ["patterns not distributed", "patterns distributed"]
for distributed in [0, 1]:
    x = results_dist_pattern[results_dist_pattern["distributed"] == distributed]["ndata"]
    y = results_dist_pattern[results_dist_pattern["distributed"] == distributed]["mean_time"]
    yerr = results_dist_pattern[results_dist_pattern["distributed"] == distributed]["std_time"]
    plt.errorbar(x, y, yerr=yerr, label=distribute_labels[distributed])
    plt.xlabel("size of the dataset")
    plt.ylabel("mean execution time [s]")
    plt.xscale("log")

plt.legend()
plt.show()
# %%
### Perf vs ndata & npattern when distributed vs not, at constant workload, NO GPU
results_dist_pattern = pd.read_csv("results_distributed_and_pattern_v4_no_gpu.csv")
distribute_labels = ["patterns not distributed", "patterns distributed"]
for distributed in [0, 1]:
    x = results_dist_pattern[results_dist_pattern["distributed"] == distributed]["ndata"]
    y = results_dist_pattern[results_dist_pattern["distributed"] == distributed]["mean_time"]
    yerr = results_dist_pattern[results_dist_pattern["distributed"] == distributed]["std_time"]
    plt.errorbar(x, y, yerr=yerr, label=distribute_labels[distributed])
    plt.xlabel("size of the dataset")
    plt.ylabel("mean execution time [s]")
    plt.xscale("log")

plt.legend()
plt.show()

# %%
