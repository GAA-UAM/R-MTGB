import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_entry import *


def scatter(x, y, task, pred, title):
    try:
        df = pd.DataFrame({"X": x.flatten(), "Y": y, "Task": task})
    except:
        df = pd.DataFrame({"X": x[:, 0], "Y": y, "Task": x[:, 1]})
    unique_tasks = df["Task"].unique()
    task_colors = {
        task: color
        for task, color in zip(
            unique_tasks,
            ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"],
        )
    }

    df["Color"] = df["Task"].map(task_colors)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["X"],
        df["Y"],
        c=df["Color"],
        edgecolor="k",
        s=100,
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=task_colors[task],
            markersize=10,
            label=task,
        )
        for task in unique_tasks
    ]
    plt.legend(
        handles=handles,
        title="Task",
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=4,
        frameon=False,
    )

    x = np.column_stack((x, task))
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    x, t = split_task(x, -1)
    unique = np.unique(t)
    T = len(unique)
    tasks_dic = dict(zip(unique, range(T)))

    for i, (r_label, r) in enumerate(tasks_dic.items()):
        idx_r = t == r_label
        X_r = x[idx_r]
        pred_r = pred[idx_r]
        if X_r.ndim > 1:
            X_r = X_r[:, 0]
        plt.scatter(
            X_r,
            pred_r,
            color=colors[i],
            label=f"task{r}",
            edgecolors="black",
            linewidths=2,
        )

    plt.title(f"{title}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the figure after everything is rendered
    plt.savefig(f"model_{title}.png", bbox_inches="tight")
    plt.show()
