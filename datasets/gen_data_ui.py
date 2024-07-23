# %%

from func_gen import FuncGen


class data_gen:
    def __init__(self, n):
        self.task_gen = FuncGen(n)

    def __call__(self, problem, scenario):

        if problem == "clf":
            tasks = self.task_gen(clf=True, scenario=scenario)
        else:
            self.task_gen(clf=False, scenario=scenario)


gen_data = data_gen(200)

for problem in ["clf", "reg"]:
    for scenario in [1, 2, 3, 4]:
        gen_data(problem, scenario)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
x = pd.read_csv('clf_4.csv')
x.drop(columns=['Unnamed: 0'], inplace=True)


data = x[x['Task']==3]



plt.scatter(data["Feature 1"], data["Feature 2"], c=data['target'])
