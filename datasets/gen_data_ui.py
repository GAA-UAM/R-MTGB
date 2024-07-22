# %%

from random_fourier import TaskGenerator


class data_gen:
    def __init__(self, n):
        self.task_gen = TaskGenerator(n)

    def __call__(self, problem, scenario):

        if problem == "clf":
            tasks = self.task_gen(clf=True, scenario=scenario)
        else:
            self.task_gen(clf=False, scenario=scenario)


gen_data = data_gen(200)

for problem in ["clf", "reg"]:
    for scenario in [1, 2, 3, 4]:
        gen_data(problem, scenario)
