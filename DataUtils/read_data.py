# %%
import os
import pandas as pd
import numpy as np

datasets = ["school", "computer", "adult", "landmine", "parkinson"]


def read_csv_safely(path):
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

def read_data(dataset):
    base_path = os.path.join(os.path.dirname(__file__), "..", "Datasets", dataset)
    
    if dataset == "adult":
        data_train = read_csv_safely(os.path.join(base_path, f"{dataset}_train_data.csv"))
        target_train = read_csv_safely(os.path.join(base_path, f"{dataset}_train_target.csv"))
        data_test = read_csv_safely(os.path.join(base_path, f"{dataset}_test_data.csv"))
        target_test = read_csv_safely(os.path.join(base_path, f"{dataset}_test_target.csv"))
        return data_train, target_train, data_test, target_test
    else:
        data = read_csv_safely(os.path.join(base_path, f"{dataset}_data.csv"))
        target = read_csv_safely(os.path.join(base_path, f"{dataset}_target.csv"))
        return data, target, _, _

X, y, _, _ =read_data("parkinson")
