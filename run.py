# %%
from model import clf
from dataset import toy_data
import matplotlib.pyplot as plt


df = toy_data(n_samples=200, n_classes=2)

X, y, task = (
    df.drop(columns=["target", "task"]).values,
    df.target.values,
    df.task.values,
)
colors = ['r', 'g', 'b', 'k', 'y']


fig, ax1 = plt.subplots(1, 2, figsize=(7, 3))
for class_label in range(len(set(y))):
    ax1[0].scatter(
        df[(df["target"] == class_label) & (df["task"] == 0)].feature_0,
        df[(df["target"] == class_label) & (df["task"] == 0)].feature_1,
        color=colors[class_label],
        label=f"original_data_class_{class_label}",
    )

# Scatter plot for noised data
for class_label in range(len(set(y))):
    ax1[1].scatter(
        df[(df["target"] == class_label) & (df["task"] == 1)].feature_0,
        df[(df["target"] == class_label) & (df["task"] == 1)].feature_1,
        color=colors[class_label],
        label=f"noised_data_class_{class_label}",
    )
ax1[1].set_title("noised_data")
ax1[0].set_title("original_data")

model = clf()
model.fit(X, y, task)
# model.predict(X)
# %%
