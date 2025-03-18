# %%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error


def report(training_set):

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
    rmse_per_task_list = []
    rmse_all_tasks_list = []
    sigmoid_theta_list = []

    for root, _, files in os.walk(os.getcwd()):
        y_test = None
        y_test_std = []
        if root.endswith("reg"):
            if not training_set:
                set_ = "y_test.csv"
                term = "test"
            else:
                set_ = "y_train.csv"
                term = "train"
            y_test_path = os.path.join(root, set_)
            sigmoid_theta_path = os.path.join(root, "sigmoid_theta.csv")
            if not os.path.exists(y_test_path):
                print(f"Skipping {root}, no y_test.csv found.")
                continue
            y_test = pd.read_csv(y_test_path, header=None)
            y_test_std.append(np.std(y_test))
            sigmoid_theta = pd.read_csv(sigmoid_theta_path, header=None)
            rmse_per_task_dict = {model: dict(zip(tasks, range(8))) for model in models}
            rmse_all_tasks_dict = {model: 0 for model in models}

            # print(sigmoid_theta)
            if np.argmax(sigmoid_theta) == 0:
                print(sigmoid_theta)
                sigmoid_theta = 1 - sigmoid_theta
                print(sigmoid_theta)
                print("--------")
            sigmoid_theta_list.append(sigmoid_theta)

            processed_models = set()
            for file_ in os.listdir(root):
                if file_.endswith(".csv") and file_.startswith("pred_"):
                    if term in file_:
                        file_name = os.path.splitext(file_)[0]  # Remove .csv extension
                        model_name = file_name.replace(
                            "pred_)", ")"
                        )  # Remove "pred_" prefix

                        model_name = model_name.replace("pred_", "")
                        model_name = model_name.replace(f"{term}_", "")

                        if model_name in models and model_name not in processed_models:
                            pred_path = os.path.join(root, file_)
                            pred = pd.read_csv(pred_path, header=None)

                            pred_t, _ = split_task(pred.values)
                            y_t, t = split_task(y_test.values)
                            pred_t = pred_t.squeeze()
                            y_t = y_t.squeeze()

                            unique_values = np.unique(pred.values[:, -1])
                            T = unique_values.size
                            task_dic = dict(zip(unique_values, range(T)))

                            rmse_all_tasks_value = root_mean_squared_error(y_t, pred_t)

                            rmse_all_tasks_dict[model_name] = rmse_all_tasks_value
                            # print(model_name, rmse_all_tasks_value)

                            for r_label, r in task_dic.items():
                                idxt = t == r
                                rmse_per_task = root_mean_squared_error(
                                    y_t[idxt], pred_t[idxt]
                                )
                                rmse_per_task_dict[model_name][
                                    f"task_{r}"
                                ] = rmse_per_task
                                # print(r, rmse_per_task)
                            processed_models.add(model_name)
            rmse_per_task_list.append(rmse_per_task_dict)
            rmse_all_tasks_list.append(rmse_all_tasks_dict)

    df_list_per_task = [
        pd.DataFrame.from_dict(rmses, orient="index") for rmses in rmse_per_task_list
    ]
    avg_df_per_task = pd.concat(df_list_per_task).groupby(level=0).agg(["mean", "std"])
    df_list_all_tasks = [
        pd.DataFrame.from_dict(rmses, orient="index") for rmses in rmse_all_tasks_list
    ]
    avg_df_all_tasks = (
        pd.concat(df_list_all_tasks).groupby(level=0).agg(["mean", "std"])
    )

    sigmoid_theta_pd = pd.DataFrame(np.mean((sigmoid_theta_list), axis=0))

    return (
        avg_df_per_task,
        avg_df_all_tasks,
        sigmoid_theta_pd,
        sigmoid_theta_list,
        np.mean(y_test_std),
    )


path = "8tasks_1outliers_5features_200_training_instances"

try:
    os.chdir(path)
except:
    pass

(
    train_df_per_task,
    train_df_all_tasks,
    sigmoid_theta_pd,
    sigmoid_theta_list,
    y_train_std,
) = report(training_set=True)

(
    test_df_per_task,
    test_df_all_tasks,
    _,
    _,
    y_test_std,
) = report(training_set=False)


test_df_per_task
test_df_all_tasks
test_df_per_task
train_df_per_task

# %%
def result_2_show(df):
    exclude_names = ["GB_datapooling", "GB_datapooling_task_as_feature", "GB_single_task"]
    df_filtered  = df[~df.index.isin(exclude_names)]
    return df_filtered

result_2_show(train_df_all_tasks)
print(y_test_std)
test_df_per_task
# %%
print(r"\sigma(\theta)")
import matplotlib.pyplot as plt

plt.plot(sigmoid_theta_pd)
plt.legend(
    # labels=[f"task {i}" for i in range(8)],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=4,
    fontsize=12,
    frameon=True,
)

plt.xlabel("Tasks", fontsize=12)
plt.ylabel("Sigmoid_theta", fontsize=12)
plt.title("average of Sigmoid_theta", fontsize=14)
plt.grid()
# %%
# np.argmax()
sigmoid_theta_pd
# %%
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 1, figsize=(8, 12))  # Creates 4 subplots in a single column

for r in range(4):
    ax[r].plot(sigmoid_theta_list[r])
    if r == 3:
        for i in range(8):
            ax[r].legend(
                labels=[f"task {i}" for i in range(8)],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=4,
                fontsize=12,
                frameon=True,
            )

    ax[r].set_xlabel("Epochs", fontsize=12)
    ax[r].set_ylabel("Sigmoid_theta", fontsize=12)
    ax[r].set_title(f"Sigmoid_theta for the experiment {r}", fontsize=14)

fig.suptitle("length_scale=0.125")
plt.tight_layout()
plt.savefig("sigmoid_theta.png")
plt.show()

# %%
test_df_per_task
