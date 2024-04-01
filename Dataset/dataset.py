import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def binary(num_instances=100, same_task=1):
    np.random.seed(42)

    X = np.random.normal(size=(num_instances, 2))

    w = -1 if not same_task else 1
    weights_task_0 = np.array([w, 1])
    weights_task_1 = np.array([1, 1])

    y_task_0 = np.dot(X, weights_task_0) > 0
    y_task_1 = np.dot(X, weights_task_1) > 0

    y_task_0 = np.where(y_task_0, 1, -1)
    y_task_1 = np.where(y_task_1, 1, -1)

    data_task_0 = pd.DataFrame(
        {"Feature 1": X[:, 0], "Feature 2": X[:, 1], "target": y_task_0, "Task": 0}
    )
    data_task_1 = pd.DataFrame(
        {"Feature 1": X[:, 0], "Feature 2": X[:, 1], "target": y_task_1, "Task": 1}
    )

    data = pd.concat([data_task_0, data_task_1], ignore_index=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    sc0 = axs[0].scatter(
        data_task_0["Feature 1"],
        data_task_0["Feature 2"],
        c=data_task_0["target"],
        cmap="viridis",
        marker="o",
        edgecolors="k",
    )
    sc1 = axs[1].scatter(
        data_task_1["Feature 1"],
        data_task_1["Feature 2"],
        c=data_task_1["target"],
        cmap="viridis",
        marker="o",
        edgecolors="k",
    )
    axs[0].set_title("task 0")
    axs[1].set_title("task 1")
    fig.suptitle("Binary Classification Dataset")

    fig.colorbar(sc0, ax=axs[0], label="target")
    fig.colorbar(sc1, ax=axs[1], label="target")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    axs[1].grid(True)
    axs[0].grid(True)
    title = "same tasks" if same_task else "different tasks"
    fig.savefig("Binary_" + title + ".png", dpi=500)
    fig.show()

    return data


def regression(num_instances=100, same_task=False):

    def generate_data(
        num_instances,
        task_num,
        same_task,
    ):
        np.random.seed(42)
        features = np.random.rand(num_instances, 2)
        if same_task:
            labels = np.sin(features[:, 0])  # Sine for task 0
        else:
            if task_num == 0:
                labels = np.sin(features[:, 0])
            else:
                labels = np.cos(features[:, 0])  # Cosine for task 1
        return pd.DataFrame(
            {
                "Feature 1": features[:, 0],
                "Feature 2": features[:, 1],
                "target": labels,
                "Task": task_num,
            }
        )

    task_zero_data = generate_data(num_instances, 0, same_task)
    task_one_data = generate_data(num_instances, 1, same_task)

    data = pd.concat([task_zero_data, task_one_data], ignore_index=True)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        task_zero_data["Feature 1"],
        task_zero_data["target"],
        color="blue",
        label="Task 0 (Sine)",
    )

    plt.scatter(
        task_one_data["Feature 1"],
        task_one_data["target"],
        color="red",
        label="Task 1 (Cosine)",
    )

    plt.title("Synthetic Regression Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("target")
    plt.legend()
    plt.grid(True)
    title = "same tasks" if same_task else "different tasks"
    plt.savefig("Regression_" + title + ".png", dpi=500)
    plt.show()

    return data
