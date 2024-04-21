import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TaskGenerator:
    def __init__(self, num_instances, seed):

        self.seed = seed
        self.N = num_instances
        self.D = 2  # Two input features
        self.W = np.random.randn(
            self.N, self.D
        )  # Random weights for the Fourier features
        self.b = np.random.uniform(
            0, 2 * np.pi, self.N
        )  # Random biases for the Fourier features
        self.theta = np.random.randn(self.N)  # Random weights for each Fourier feature
        self.alpha = 100.0
        self.l = 10.0

    def _f(self, x):
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

    def _target_cs(self, x_c, x_s):
        return np.concatenate((self._f(x_c), self.specific_part(x_s)))

    def _target_s(self, x_c, x_s):
        return np.concatenate((self.specific_part(x_c), self.specific_part(x_s)))

    def _target_c(self, x_c, x_s):
        return np.concatenate((self._f(x_c), self._f(x_s)))

    def specific_part(self, x):
        return np.sin(x[:, 0]) * np.cos(x[:, 1])

    def _task_gen(self, clf):

        t1 = np.random.default_rng(np.random.randint(self.seed))
        x_common = t1.uniform(-3, 3, size=(self.N, self.D))

        t2 = np.random.default_rng(np.random.randint(self.seed))
        x_specific = t2.uniform(-3, 3, size=(self.N, self.D))

        x_cs = np.concatenate((x_common, x_specific))
        x_s = np.concatenate((x_specific, x_specific))
        x_c = np.concatenate((x_common, x_common))

        if clf:
            # Generate target values for each task
            y_cs = self._classify_output(self._target_cs(x_common, x_specific))
            y_s = self._classify_output(self._target_s(x_specific, x_specific))
            y_c = self._classify_output(self._target_c(x_common, x_common))
        else:
            y_cs = self._target_cs(x_common, x_specific)
            y_s = self._target_s(x_specific, x_specific)
            y_c = self._target_c(x_common, x_common)

        return (
            x_c,
            y_c,
            x_s,
            y_s,
            x_cs,
            y_cs,
        )

    def _classify_output(self, output):
        threshold = np.mean(output)
        return np.where(output > threshold, 1, 0)

    def clf(self):
        x_c, y_c, x_s, y_s, x_cs, y_cs = self._task_gen(True)
        self._plot(x_c, y_c, x_s, y_s, x_cs, y_cs, "classification")
        dfs = []
        for i, (x, y) in enumerate([(x_c, y_c), (x_s, y_s), (x_cs, y_cs)]):
            dfs.append(self._gen_df(x, y, i))
        return pd.concat(dfs, ignore_index=True)

    def reg(self):
        x_c, y_c, x_s, y_s, x_cs, y_cs = self._task_gen(False)
        self._plot(x_c, y_c, x_s, y_s, x_cs, y_cs, "Regression")
        dfs = []
        for i, (x, y) in enumerate([(x_c, y_c), (x_s, y_s), (x_cs, y_cs)]):
            dfs.append(self._gen_df(x, y, i))
        return pd.concat(dfs, ignore_index=True)

    def _gen_df(self, x, y, task_num):
        df = pd.DataFrame(
            np.column_stack((x, y)), columns=["Feature 1", "Feature 2", "target"]
        )
        df["Task"] = np.ones_like(y) * task_num
        return df

    def _plot(
        self,
        x_c,
        y_c,
        x_s,
        y_s,
        x_cs,
        y_cs,
        title,
    ):

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].scatter(x_c[:, 0], x_c[:, 1], c=y_c, cmap="viridis")
        axs[0].set_xlabel("x1")
        axs[0].set_ylabel("x2")
        axs[0].set_title("Common and specific part")

        axs[1].scatter(x_s[:, 0], x_s[:, 1], c=y_s, cmap="viridis")
        axs[1].set_xlabel("x1")
        axs[1].set_ylabel("x2")
        axs[1].set_title("Specific part")

        axs[2].scatter(x_cs[:, 0], x_cs[:, 1], c=y_cs, cmap="viridis")
        axs[2].set_xlabel("x1")
        axs[2].set_ylabel("x2")
        axs[2].set_title("Common part")

        plt.colorbar(
            axs[0].scatter(
                x_c[:, 0],
                x_c[:, 1],
                c=y_c,
                cmap="viridis",
            ),
            ax=axs[0],
            label="Class",
        )
        plt.colorbar(
            axs[1].scatter(
                x_s[:, 0],
                x_s[:, 1],
                c=y_s,
                cmap="viridis",
            ),
            ax=axs[1],
            label="Class",
        )
        plt.colorbar(
            axs[2].scatter(
                x_cs[:, 0],
                x_cs[:, 1],
                c=y_cs,
                cmap="viridis",
            ),
            ax=axs[2],
            label="Class",
        )

        fig.suptitle(f"{title}")

        fig.tight_layout()
        fig.savefig(f"{title}.png")
        plt.show()
