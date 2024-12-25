# %%
import os
import numpy as np
from func_gen import GenerateDataset
from sklearn.model_selection import train_test_split


class data_gen:
    def __init__(self, scenario):
        self.scenario = scenario

    def __call__(self, regression):
        self.task_gen = GenerateDataset(self.scenario)
        return self.task_gen(regression)


if __name__ == "__main__":

    base_dir = "8tasks_3outliers"

    if os.path.exists(base_dir):
        print("a")
    else:
        os.mkdir(base_dir)

    original_dir = os.getcwd()

    # for scenario in [1, 2, 3, 4]:
    for scenario in [4]:
        for i, subdir in enumerate(range(1, 10 + 1)):
            scenario_name = "scenario_" + str(scenario)
            dir_path = os.path.join(base_dir, scenario_name, str(subdir))
            os.makedirs(dir_path, exist_ok=True)
            os.chdir(dir_path)
            os.mkdir("clf")
            os.mkdir("reg")
            # for regression in [True, False]:
            for regression in [True]:
                gen_data2 = data_gen(scenario)
                df = gen_data2(regression)
                train_df, test_df = train_test_split(
                    df,
                    train_size=0.1,
                    random_state=np.random.randint(i, i + 1),
                    stratify=None if regression else df.target,
                )
                train_df.to_csv(
                    f"train_{'reg' if regression else 'clf'}_{scenario}.csv"
                )
                test_df.to_csv(f"test_{'reg' if regression else 'clf'}_{scenario}.csv")
            os.chdir(original_dir)
        os.chdir(original_dir)

# %%
