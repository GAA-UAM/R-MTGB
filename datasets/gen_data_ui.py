# %%
import os
import pandas as pd
from func_gen import GenerateDataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    original_dir = os.getcwd()
    num_tasks = 8
    num_dims = 8
    num_train_per_task = 50
    num_test_per_task = 1000
    num_outlier_tasks = 6
    num_instances = num_train_per_task + num_test_per_task

    base_dir = f"{num_tasks}tasks_{num_outlier_tasks}outliers_{num_dims}features_{num_train_per_task}training"

    if os.path.exists(base_dir):
        print("a")
    else:
        os.mkdir(base_dir)

    # for scenario in [1, 2, 3, 4]:
    for scenario in [4]:
        for i, subdir in enumerate(range(1, 100 + 1)):
            scenario_name = "scenario_" + str(scenario)
            # dir_path = os.path.join(base_dir, scenario_name, str(subdir))
            dir_path = os.path.join(base_dir, str(subdir))
            os.makedirs(dir_path, exist_ok=True)
            os.chdir(dir_path)
            os.mkdir("clf")
            os.mkdir("reg")
            # for regression in [True, False]:
            for regression in [True]:
                gen_data = GenerateDataset(scenario)
                df = gen_data(
                    regression=regression,
                    num_dims=num_dims,
                    num_tasks=num_tasks,
                    num_instances=num_instances,
                    num_outlier_tasks=num_outlier_tasks,
                )

                train_dfs, test_dfs = [], []
                for task_id in df["Task"].unique():
                    task_df = df[df["Task"] == task_id]

                    train_task_df, test_task_df = train_test_split(
                        task_df,
                        train_size=num_train_per_task,
                        test_size=num_test_per_task,
                        random_state=int(i),
                    )

                    train_dfs.append(train_task_df)
                    test_dfs.append(test_task_df)

                    # Concatenate all tasks
                    train_df = pd.concat(train_dfs)
                    test_df = pd.concat(test_dfs)

                train_df.to_csv(
                    f"train_{'reg' if regression else 'clf'}_{scenario}.csv"
                )
                test_df.to_csv(f"test_{'reg' if regression else 'clf'}_{scenario}.csv")
            os.chdir(original_dir)
        os.chdir(original_dir)

# %%
import numpy as np

np.std(df["Target"])
