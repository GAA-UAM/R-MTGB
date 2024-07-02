import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class TaskGenerator:
    def __init__(self):

        self.N = (
            500  # Number of features, only to generate the fun output not the X size.
        )
        self.D = 2  # Number of features (dimension)
        self.W = np.random.randn(
            self.N, self.D
        )  # Random weights for the Fourier features
        self.b = np.random.uniform(
            0, 2 * np.pi, self.N
        )  # Random biases for the Fourier features
        self.theta = np.random.randn(
            self.N
        )  # Random weights for each Fourier feature, randomnes for various functions.
        self.alpha = 100  # Determines dataset size.
        self.l = 10.0

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(self.theta, bins=50)
        ax[0].set_xlabel("Theta Values")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Distribution of Theta")
        ax[1].hist(self.b, bins=50)
        ax[1].set_xlabel("Bias Values")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("Distribution of Biases")
        fig.savefig(f"variables.png")

    def _target_c(self, x):
        x = np.atleast_2d(x)

        return np.apply_along_axis(
            lambda x: np.sum(
                self.theta
                * np.sqrt(self.alpha / self.N)
                * np.cos(np.dot(self.W, x / self.l) + self.b)
            ),
            1,
            x,
        )

    def _target_s(self, x):
        return np.sin(x[:, 0]) + np.cos(x[:, 1])

    def __call__(self, clf, num_instances):

        random_seed = np.random.randint(1, 100)
        t1 = np.random.default_rng(np.random.randint(random_seed))
        x1 = t1.uniform(-3, 3, size=(num_instances, self.D))

        random_seed = np.random.randint(101, 200)
        t2 = np.random.default_rng(np.random.randint(random_seed))
        x2 = t2.uniform(-3, 3, size=(num_instances, self.D))

        random_seed = np.random.randint(201, 300)
        t3 = np.random.default_rng(np.random.randint(random_seed))
        x3 = t3.uniform(-3, 3, size=(num_instances, self.D))

        random_seed = np.random.randint(301, 400)
        t4 = np.random.default_rng(np.random.randint(random_seed))
        x4 = t4.uniform(-3, 3, size=(num_instances, self.D))

        scl = StandardScaler()
        x1 = scl.fit_transform(x1)
        x2 = scl.fit_transform(x2)
        x3 = scl.fit_transform(x3)
        x4 = scl.fit_transform(x4)

        # Generate target values for each task

        if clf:
            y1 = self._classify_output(self._multi_task_gen(x1, 0.9))
            y2 = self._classify_output(self._multi_task_gen(x2, 0.9))
            y3 = self._classify_output(self._multi_task_gen(x3, 0.9))
            y4 = self._classify_output(self._target_s(x4))

            self._plot(x1, y1, x2, y2, x3, y3, x4, y4, "classification")

        else:
            y1 = self._multi_task_gen(x1, 0.9)
            y2 = self._multi_task_gen(x2, 0.9)
            y3 = self._multi_task_gen(x3, 0.9)
            y4 = self._target_s(x4)

            self._plot(x1, y1, x2, y2, x3, y3, x4, y4, "Regression")

        dfs = []
        ranges = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        for i, (x, y) in enumerate(ranges):
            dfs.append(self._gen_df(x, y, i))

        return pd.concat(dfs, ignore_index=True)

    def _multi_task_gen(self, x, w):
        return w * self._target_c(x) + (1 - w) * self._target_s(x)

    def _classify_output(self, output):
        normalized_output = (output - np.mean(output)) / np.std(output)
        threshold = 0
        return np.where(normalized_output < threshold, 0, 1)

    def _gen_df(self, x, y, task_num):
        df = pd.DataFrame(
            np.column_stack((x, y)), columns=["Feature 1", "Feature 2", "target"]
        )
        df["Task"] = np.ones_like(y) * task_num
        return df

    def _plot(
        self,
        x1,
        y1,
        x2,
        y2,
        x3,
        y3,
        x4,
        y4,
        title,
    ):

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0][0].scatter(x1[:, 0], x1[:, 1], c=y1, cmap="viridis")
        axs[0][0].set_xlabel("Feature 1")
        axs[0][0].set_ylabel("Feature 2")
        axs[0][0].set_title("Task 1")

        axs[0][1].scatter(x2[:, 0], x2[:, 1], c=y2, cmap="viridis")
        axs[0][1].set_xlabel("Feature 1")
        axs[0][1].set_ylabel("Feature 2")
        axs[0][1].set_title("Task 2")

        axs[1][0].scatter(x3[:, 0], x3[:, 1], c=y3, cmap="viridis")
        axs[1][0].set_xlabel("Feature 1")
        axs[1][0].set_ylabel("Feature 2")
        axs[1][0].set_title("Task 3")

        axs[1][1].scatter(x4[:, 0], x4[:, 1], c=y4, cmap="viridis")
        axs[1][1].set_xlabel("Feature 1")
        axs[1][1].set_ylabel("Feature 2")
        axs[1][1].set_title("Task 4")

        plt.colorbar(
            axs[0][0].scatter(
                x1[:, 0],
                x1[:, 1],
                c=y1,
                cmap="viridis",
            ),
            ax=axs[0][0],
            label="target",
        )
        plt.colorbar(
            axs[0][1].scatter(
                x2[:, 0],
                x2[:, 1],
                c=y2,
                cmap="viridis",
            ),
            ax=axs[0][1],
            label="target",
        )
        plt.colorbar(
            axs[1][0].scatter(
                x3[:, 0],
                x3[:, 1],
                c=y3,
                cmap="viridis",
            ),
            ax=axs[1][0],
            label="target",
        )
        plt.colorbar(
            axs[1][1].scatter(
                x4[:, 0],
                x4[:, 1],
                c=y4,
                cmap="viridis",
            ),
            ax=axs[1][1],
            label="target",
        )

        fig.suptitle(f"{title}")

        fig.tight_layout()
        fig.savefig(f"{title}.png")
        plt.show()
