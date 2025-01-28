import pandas as pd
import numpy as np


def split_task(X, task_info):

    unique_values = np.unique(X[:, task_info])
    mapping = {value: index for index, value in enumerate(unique_values)}
    X[:, task_info] = [mapping[value] for value in X[:, task_info]]

    X_task = X[:, task_info]
    X_data = np.delete(X, task_info, axis=1).astype(float)

    return X_data, X_task


def data():

    df = pd.read_csv(r"tasks_data_128_25_instances.csv")
    x_train, y_train, task_train = df["X"].values, df["Y"].values, df["Task"].values
    x_train = x_train.reshape(-1, 1)

    return x_train, y_train, task_train
