# %%
import os
import pandas as pd
from pathlib import Path
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("seaborn-v0_8-whitegrid")


def read_csv(dataset, model):
    y_test = y_train = pred_test = pred_train = None
    current_file_path = Path(__file__).resolve()
    script_dir = current_file_path.parent
    for root, _, files in os.walk(script_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".csv") and dataset in file:
                if fnmatch.fnmatch(file, f"*sigmoid_theta_{model}*.csv"):
                    arr = pd.read_csv(file_path, header=None).values
                    return arr
    return None


sigmoid_thetas = {"MTB": [], "POOLING": [], "RMTB": [], "STL": []}
datasets = ["school", "computer", "parkinson", "landmine", "adult_gender", "adult_race", "bank", "avila", "abalone", "sarcos"]
sigmoid_thetas = {dataset: [] for dataset in datasets}

for dataset in datasets:
    arr = read_csv(dataset, "RMTB")
    if arr is not None:
        sigmoid_thetas[dataset].append({"dataset": dataset, "values": arr})


fig, axs = plt.subplots(5, 2, figsize=(14, 10))
axs = axs.flatten()

all_y_values = []
for i, dataset in enumerate(datasets):
    ax = axs[i]
    values = [entry["values"] for entry in sigmoid_thetas[dataset]]
    if values:
        y = values[0].flatten()
        all_y_values.extend(y)

y_min = min(all_y_values)
y_max = max(all_y_values)

for i, dataset in enumerate(datasets):
    ax = axs[i]
    values = [entry["values"] for entry in sigmoid_thetas[dataset]]
    if values:
        y = values[0].flatten()
        x = np.arange(len(y))
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, color="#007ACC")

        ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Task ID", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        
        ax.set_ylim(y_min, y_max)

        # Reduce number of x-ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_title(f"{dataset} (No data)")
        ax.axis("off")

# Hide unused subplots
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.suptitle("Sigmoid Thetas Across Datasets", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("sigmoid_thetas.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
