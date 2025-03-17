# %%
import os
from func_gen import GenerateDataset
from sklearn.model_selection import train_test_split


class data_gen:
    def __init__(self, scenario):
        self.scenario = scenario

    def __call__(self, regression, n_dim, n_tasks, n_instances):
        self.task_gen = GenerateDataset(self.scenario)
        return self.task_gen(regression, n_dim, n_tasks, n_instances)


if __name__ == "__main__":

    base_dir = "8tasks_1outliers_5features_10_training_instances"

    if os.path.exists(base_dir):
        print("a")
    else:
        os.mkdir(base_dir)

    original_dir = os.getcwd()

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
                gen_data = data_gen(scenario)
                df = gen_data(regression, 5, 8, 1000)
                train_df, test_df = train_test_split(
                    df, test_size=0.99, random_state=42
                )
                train_df.to_csv(
                    f"train_{'reg' if regression else 'clf'}_{scenario}.csv"
                )
                test_df.to_csv(f"test_{'reg' if regression else 'clf'}_{scenario}.csv")
            os.chdir(original_dir)
        os.chdir(original_dir)

# %%
import numpy as np

np.std(df["Target"])
