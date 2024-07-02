# %%

from toy_dataset import *
from MT_RandomFourier import TaskGenerator

# seed = 1112546
# regre = regression(num_instances=100, same_task=False, seed=seed)
# regre.to_csv("two_task_reg.csv")

# clf = binary(num_instances=100, same_task=False, seed=seed)
# clf.to_csv("two_task_clf.csv")

# regre = regression(num_instances=100, same_task=True, seed=seed)
# regre.to_csv("one_task_reg.csv")

# clf = binary(num_instances=100, same_task=True, seed=seed)
# clf.to_csv("one_task_clf.csv")

# for i in range(10):

#     task_gen = TaskGenerator(num_instances=100, tasks=None)

#     task_gen.clf().to_csv(f"one_task_clf{i}.csv")
#     task_gen.reg().to_csv(f"one_task_reg{i}.csv")

task_gen = TaskGenerator()
#%%
task_gen(clf=True, num_instances=100).to_csv(f"multi_task_clf.csv")
task_gen(clf=False, num_instances=100).to_csv(f"multi_task_reg.csv")
