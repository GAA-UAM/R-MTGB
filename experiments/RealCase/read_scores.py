# %%
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    root_mean_squared_error,
    recall_score,
    mean_absolute_error,
)
from collections import defaultdict
import re
from pathlib import Path
import fnmatch


def calculate_per_task_metrics(y, pred, task_type="regression"):
    task_metrics = defaultdict(list)
    task_indices = np.unique(y[:, 1])

    for task_id in task_indices:
        y_task = y[y[:, 1] == task_id][:, 0]
        pred_task = pred[pred[:, 1] == task_id][:, 0]

        if task_type == "regression":
            rmse = root_mean_squared_error(y_task, pred_task)
            task_metrics["task_id"].append(task_id)
            task_metrics["rmse"].append(rmse)
        else:
            acc = accuracy_score(y_task, pred_task)
            task_metrics["task_id"].append(task_id)
            task_metrics["accuracy"].append(acc)

    return pd.DataFrame(task_metrics)


def read_csv(dataset, model):
    y_test = y_train = pred_test = pred_train = None
    current_file_path = Path(__file__).resolve()
    script_dir = current_file_path.parent
    for root, _, files in os.walk(script_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".csv") and dataset in file:
                if "y_test" in file:
                    y_test = pd.read_csv(file_path, header=None).values
                elif "y_train" in file:
                    y_train = pd.read_csv(file_path, header=None).values
                elif fnmatch.fnmatch(file, f"*_{model}*.csv"):
                    if "pred_test" in file:
                        pred_test = pd.read_csv(file_path, header=None).values
                    elif "pred_train" in file:
                        pred_train = pd.read_csv(file_path, header=None).values

    if all(v is not None for v in [y_test, pred_test, pred_train]) and dataset in [
        "school",
        "computer",
        "parkinson",
        "abalone",
        "sarcos",
    ]:
        rmse_test = root_mean_squared_error(y_test[:, 0], pred_test[:, 0])
        rmse_train = root_mean_squared_error(y_train[:, 0], pred_train[:, 0])

        mae_train = mean_absolute_error(y_train[:, 0], pred_train[:, 0])
        mae_test = mean_absolute_error(y_test[:, 0], pred_test[:, 0])

        results = {
            "dataset": dataset,
            "model": model,
            "task": "regression",
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "mae_train": mae_train,
            "mae_test": mae_test,
        }

        if not model == "POOLING":

            df_test_per_task = calculate_per_task_metrics(
                y_test, pred_test, task_type="regression"
            )

            df_test_per_task["dataset"] = dataset
            df_test_per_task["model"] = model
        elif model == "POOLING":
            df_test_per_task = 0.0

    elif all(v is not None for v in [y_test, pred_test, pred_train]) and dataset in [
        "adult_gender",
        "adult_race",
        "landmine",
        "bank",
        "avila",
    ]:
        acc_test = accuracy_score(y_test[:, 0], pred_test[:, 0])
        acc_train = accuracy_score(y_train[:, 0], pred_train[:, 0])

        recall_test = recall_score(y_test[:, 0], pred_test[:, 0], average="macro")
        recall_train = recall_score(y_train[:, 0], pred_train[:, 0], average="macro")

        results = {
            "dataset": dataset,
            "model": model,
            "task": "classification",
            "accuracy_train": acc_train,
            "accuracy_test": acc_test,
            "recall_train": recall_train,
            "recall_test": recall_test,
        }

        if not model == "POOLING":
            df_test_per_task = calculate_per_task_metrics(
                y_test, pred_test, task_type="classification"
            )

            df_test_per_task["dataset"] = dataset
            df_test_per_task["model"] = model
        elif model == "POOLING":
            df_test_per_task = 0.0
    else:
        results = None
        df_test_per_task = None

    return results, df_test_per_task


all_scores = []
all_task_scores = []

for model in ["MTB", "POOLING", "RMTB", "STL"]:
    for dataset in [
        "school",
        "computer",
        "parkinson",
        "landmine",
        "adult_gender",
        "adult_race",
        "bank",
        "avila",
        "abalone",
        "sarcos",
    ]:
        result, df_test_per_task = read_csv(dataset, model)
        if result is not None:
            all_scores.append(result)
        if df_test_per_task is not None:
            all_task_scores.append(df_test_per_task)

# Overall scores
df_scores = pd.DataFrame(all_scores)
df_scores.to_csv("scores.csv", index=False)

df_task_scores = pd.concat(
    [
        df
        for df in all_task_scores
        if isinstance(df, pd.DataFrame) and not df.equals(0.0)
    ],
    ignore_index=True,
)
# %%
# Task ranking
df_task_scores["rmse_rank"] = df_task_scores.groupby(["dataset", "task_id"])[
    "rmse"
].rank(ascending=True, method="average", axis=0)
df_task_scores["accuracy_rank"] = df_task_scores.groupby(["dataset", "task_id"])[
    "accuracy"
].rank(ascending=False, method="average", axis=0)
df_task_scores.to_csv("scores_per_task.csv", index=False)


df_task_scores["task_key"] = (
    df_task_scores["dataset"].astype(str) + "_" + df_task_scores["task_id"].astype(str)
)
is_classification = df_task_scores["accuracy"].notnull()
is_regression = df_task_scores["rmse"].notnull()
clf_df = df_task_scores[is_classification].copy()
clf_df["rank"] = clf_df.groupby("task_key")["accuracy"].rank(ascending=False)
clf_pivot = clf_df.pivot(index="model", columns="task_key", values="rank")
reg_df = df_task_scores[is_regression].copy()
reg_df["rank"] = reg_df.groupby("task_key")["rmse"].rank(ascending=True)
reg_pivot = reg_df.pivot(index="model", columns="task_key", values="rank")
clf_pivot = clf_pivot.sort_index(axis=1).sort_index().T.to_csv("clf_pivot.csv")
reg_pivot = reg_pivot.sort_index(axis=1).sort_index().T.to_csv("reg_pivot.csv")
