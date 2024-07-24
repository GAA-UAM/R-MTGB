# %%
import os
from func_gen2 import GenerateDataset


class data_gen:
    def __init__(self, scenario):
        self.scenario = scenario

    def __call__(self, regression, test):
        self.task_gen = GenerateDataset(self.scenario)
        self.task_gen(regression, test)


if __name__ == "__main__":

    base_dir = "24Jul"

    if os.path.exists(base_dir):
        print("a")
    else:
        os.mkdir(base_dir)

    original_dir = os.getcwd()

    for scenario in [1, 2, 3, 4]:
        for subdir in range(1, 10 + 1):
            scenario_name = "scenario_" + str(scenario)
            dir_path = os.path.join(base_dir, scenario_name, str(subdir))
            os.makedirs(dir_path, exist_ok=True)
            os.chdir(dir_path)
            os.mkdir("clf")
            os.mkdir("reg")
            for regression in [True, False]:
                gen_data2 = data_gen(scenario)
                gen_data2(regression, False)
            os.chdir(original_dir)
        dir_path = os.path.join(base_dir, scenario_name, "test_data")
        os.makedirs(dir_path, exist_ok=True)
        os.chdir(dir_path)
        for regression in [True, False]:
            gen_data2 = data_gen(scenario)
            gen_data2(regression, True)
        os.chdir(original_dir)
