# %%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def split_task(X):
    unique_values = np.unique(X[:, -1])
    mapping = {value: index for index, value in enumerate(unique_values)}
    X[:, -1] = np.vectorize(mapping.get)(X[:, -1])

    X_t = X[:, -1]
    X_d = np.delete(X, -1, axis=1).astype(np.float64)
    return X_d, X_t


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
mse_list = []


for root, _, files in os.walk(os.getcwd()):
    for file_ in files:
        if file_.endswith(".csv"):
            file_path = os.path.join(root, file_)

            if "y_test" in file_:
                y_test = pd.read_csv(file_path, header=None)

            matched_model = next((kw for kw in models if kw in file_), None)

            if matched_model:
                pred = pd.read_csv(file_path, header=None)

                pred_t, t = split_task(pred.values)
                y, _ = split_task(y_test.values)

                unique_values = np.unique(t)
                T = unique_values.size
                task_dic = dict(zip(unique_values, range(T)))

                for r_label, r in task_dic.items():
                    mse_value = mean_squared_error(y[t == r], pred_t[t == r])
                    mse[matched_model][f"task_{r}"] = mse_value
