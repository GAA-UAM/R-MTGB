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


def read_csv(dataset, model):
    y_test = y_train = pred_test = pred_train = None
    current_file_path = Path(__file__).resolve()
    script_dir = current_file_path.parent
    for root, _, files in os.walk(script_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".csv") and dataset in file:
                if fnmatch.fnmatch(file, f"*sigmoid_theta_{model}*.csv"):
                    df = pd.read_csv(file_path, header=None).values
                    return df
    return None


sigmoid_thetas = {"MTB": [], "POOLING": [], "RMTB": [], "STL": []}


for model in ["MTB", "POOLING", "RMTB", "STL"]:
    for dataset in [
        "school",
        "computer",
        "parkinson",
        "landmine",
        "adult_gender",
        "adult_race",
    ]:
        df = read_csv(dataset, model)
        if df is not None:
            sigmoid_thetas[model].append({"dataset": dataset, "values": df})

