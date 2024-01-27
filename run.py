# %%
from model import clf
from dataset import toy_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df = toy_data(n_samples=200)

X, y, task = (
    df.drop(columns=["target", "task"]).values,
    df.target.values,
    df.task.values,
)
stack = np.column_stack((X, y, task))



# fig, ax1 = plt.subplots(1, 2, figsize=(7, 3))
# ax1[0].scatter(
#     (df[df["task"] == 0]).feature_0,
#     (df[df["task"] == 0]).feature_1,
#     color="r",
#     label="original_data",
# )
# ax1[1].scatter(
#     (df[df["task"] == 1]).feature_0,
#     (df[df["task"] == 1]).feature_1,
#     label="noised_data",
# )
# ax1[1].set_title("noised_data")
# ax1[0].set_title("original_data")


model = clf()
model.fit(X, y, task)


# %%
