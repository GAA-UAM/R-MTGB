# %%
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import root_mean_squared_error, accuracy_score


def report(training_set, regression=True):
    if regression:
        problem = "reg"
        scoring_func = root_mean_squared_error
    else:
        problem = "clf"
        scoring_func = accuracy_score

    def split_task(X):
        unique_values = np.unique(X[:, -1])
        mapping = {value: index for index, value in enumerate(unique_values)}
        X[:, -1] = np.vectorize(mapping.get)(X[:, -1])

        X_t = X[:, -1]
        X_d = np.delete(X, -1, axis=1).astype(np.float64)
        return X_d, X_t

    models = [
        "MTB",
        "POOLING",
        "POOLING_TASK_AS_FEATURE",
        "STL",
        "RMTB",
    ]

    tasks = [f"task_{i}" for i in range(8)]
    score_per_task_list = []
    score_all_tasks_list = []
    sigmoid_theta_list = []
    sigmoid_all_tasks_list = []

    for root, _, files in os.walk(os.getcwd()):
        y_test = None
        y_test_std = []
        if root.endswith(problem):
            if not training_set:
                set_ = "y_test.csv"
                term = "test"
            else:
                set_ = "y_train.csv"
                term = "train"
            y_test_path = os.path.join(root, set_)

            if not os.path.exists(y_test_path):
                continue
            y_test = pd.read_csv(y_test_path, header=None)
            y_test = y_test.fillna(0)
            y_test_std.append(np.std(y_test))
            score_per_task_dict = {
                model: dict(zip(tasks, range(8))) for model in models
            }
            score_all_tasks_dict = {model: 0 for model in models}

            processed_models = set()
            for file_ in os.listdir(root):
                if file_.endswith(".csv") and file_.startswith("pred_"):
                    if term in file_:
                        file_name = os.path.splitext(file_)[0]  # Remove .csv extension
                        model_name = file_name.replace("pred_)", ")")

                        model_name = model_name.replace("pred_", "")
                        model_name = model_name.replace(f"{term}_", "")

                        if model_name in models and model_name not in processed_models:
                            pred_path = os.path.join(root, file_)
                            pred = pd.read_csv(pred_path, header=None)

                            pred = pred.fillna(0)

                            pred_t, _ = split_task(pred.values)
                            y_t, t = split_task(y_test.values)
                            pred_t = pred_t.squeeze()
                            y_t = y_t.squeeze()

                            unique_values = np.unique(pred.values[:, -1])
                            T = unique_values.size
                            task_dic = dict(zip(unique_values, range(T)))

                            try:
                                score_all_tasks_value = scoring_func(y_t, pred_t)
                            except:
                                pred_t = pred_t[:-1]
                                score_all_tasks_value = scoring_func(y_t, pred_t)


                            score_all_tasks_dict[model_name] = score_all_tasks_value
                            if model_name == "RMTB":
                                sigmoid_theta_path = os.path.join(
                                    root, f"sigmoid_theta_{model_name}.csv"
                                )
                                sigmoid_theta = pd.read_csv(
                                    sigmoid_theta_path, header=None
                                )
                                if np.argmax(sigmoid_theta) == 0:
                                    sigmoid_theta = 1 - sigmoid_theta
                                sigmoid_theta_list.append(sigmoid_theta)
                            for r_label, r in task_dic.items():
                                idxt = t == r
                                rmse_per_task = scoring_func(y_t[idxt], pred_t[idxt])
                                score_per_task_dict[model_name][
                                    f"task_{r}"
                                ] = rmse_per_task
                            processed_models.add(model_name)
            score_per_task_list.append(score_per_task_dict)
            score_all_tasks_list.append(score_all_tasks_dict)
    df_list_per_task = [
        pd.DataFrame.from_dict(rmses, orient="index") for rmses in score_per_task_list
    ]
    avg_df_per_task = pd.concat(df_list_per_task).groupby(level=0).agg(["mean", "std"])
    df_list_all_tasks = [
        pd.DataFrame.from_dict(rmses, orient="index") for rmses in score_all_tasks_list
    ]
    avg_df_all_tasks = (
        pd.concat(df_list_all_tasks).groupby(level=0).agg(["mean", "std"])
    )

    return (
        avg_df_per_task,
        avg_df_all_tasks,
        pd.DataFrame(np.mean((sigmoid_theta_list), axis=0)),
        np.mean(y_test_std),
    )


path = r"D:\Ph.D\Programming\Py\NoiseAwareBoost\experiments\toy_data_comparison\10tasks_2outliers_5features_300training"

try:
    os.chdir(path)
except:
    pass
# %% Regression
(
    train_df_per_task,
    train_df_all_tasks,
    sigmoid_theta,
    y_train_std,
) = report(training_set=True, regression=True)

(
    test_df_per_task,
    test_df_all_tasks,
    sigmoid_theta,
    y_test_std,
) = report(training_set=False, regression=True)


# %%

# (
#     train_df_per_task,
#     train_df_all_tasks,
#     sigmoid_theta,
#     y_train_std,
# ) = report(training_set=True, regression=False)

(
    test_df_per_task,
    test_df_all_tasks,
    sigmoid_theta,
    y_test_std,
) = report(training_set=False, regression=False)


# %%
def result_2_show(df):
    exclude_names = ["POOLING", "POOLING_TASK_AS_FEATURE"]
    df_filtered = df[~df.index.isin(exclude_names)]
    return df_filtered


test_df_all_tasks
# result_2_show(train_df_per_task  )
# %%
print(r"\sigma(\theta)")

import matplotlib.pyplot as plt

plt.plot(sigmoid_theta, label="class 0")
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
