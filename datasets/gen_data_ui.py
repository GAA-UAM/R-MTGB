# %%
import os
import numpy as np
from func_gen import GenerateDataset


class data_gen:
    def __init__(self, scenario ):
        self.scenario = scenario


    def __call__(self, regression, n_dim, n_tasks, n_instances):
        self.task_gen = GenerateDataset(self.scenario)
        return self.task_gen(regression, n_dim, n_tasks, n_instances)


if __name__ == "__main__":

    base_dir = "8tasks_1outliers_5features_200_training_instances_length_scale0.125"

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
                train_df = gen_data(regression, 5, 8, 200)
                test_df = gen_data(regression, 5, 8, 300)
                train_df.to_csv(
                    f"train_{'reg' if regression else 'clf'}_{scenario}.csv"
                )
                test_df.to_csv(f"test_{'reg' if regression else 'clf'}_{scenario}.csv")
            os.chdir(original_dir)
        os.chdir(original_dir)

# %%
