# %%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

pred = pd.read_csv("pred_Proposed MT.csv", header=None)
y = pd.read_csv("y_test.csv", header=None)


def split_task(X):
    unique_values = np.unique(X[:, -1])
    mapping = {value: index for index, value in enumerate(unique_values)}
    X[:, -1] = np.vectorize(mapping.get)(X[:, -1])

    X_t = X[:, -1]
    X_d = np.delete(X, -1, axis=1).astype(np.float64)
    return X_d, X_t


pred_t, t = split_task(pred.values)
y_t, _ = split_task(y.values)

mean_squared_error(y_t, pred_t)

unique_values = np.unique(y_t)
T = unique_values.size
task_dic = dict(zip(unique_values, range(T)))
models = [
    "Conventional MT",
    "GB_datapooling",
    "GB_datapooling_task_as_feature",
    "GB_single_task",
    "Proposed MT",
]
tasks = [f"task_{i}" for i in range(8)]
mse = dict(zip([f"task_{i}" for i in range(8)], list(range(8))))
mse = {model: dict(zip(tasks, range(8))) for model in models}
pd.DataFrame(mse)
#%%

for r_label, r in task_dic.items():
    mean_squared_error(y_t[t == r], pred_t[t == r])
    print(r)
