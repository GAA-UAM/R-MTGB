import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FuncGen:
    """Function generator

    Output
    ----------

    This class creates various sets of toy datasets
    using different scenarios
    (Including 4 tasks in each setting):

        Scenario 1:
                    Generates a set of common tasks with identical
                    weights and randomness in their Fourier function
                    generator.
        Scenario 2:
                    Produces a set of specific tasks using a simple
                    trigonometric function.
        Scenario 3:
                    Utilizes the function from Scenario 1
                    but with varying weights and randomness to
                    create diverse tasks.
        Scenario 4:
                    Includes three common tasks with the
                    same parameters and one outlier task with
                    a lower participation weight.

    Parameters
    ----------
        num_instances: int
                The number of output instances.
        N: int
                The number of features, to generate the output.
        d: int
                The number of features (dimension).
        w: np.ndaaray
                Random weights for the Fourier features.
        b: np.ndaaray
                Random biases for the Fourier features.
        theta: np.ndaaray
                Coefficients to scale the cosine.
        alpha: int
                Random Fourier size control.

    Attributes
    ----------
    """

    def __init__(self, num_instances):

        self.N = 500
        self.d = 2
        self.w = np.random.randn(self.N, self.d)
        self.b = np.random.uniform(0, 2 * np.pi, self.N)
        self.theta = np.random.randn(self.N)
        self.alpha = 100
        self.l = 10.0
        self.n = num_instances

        def generate_data(seed_range, num_samples):
            random_seed = np.random.randint(seed_range[0], seed_range[1])
            rng = np.random.default_rng(np.random.randint(random_seed))
            data = rng.uniform(-3, 3, size=(num_samples, self.d))
            return StandardScaler().fit_transform(data)

        self.x1 = generate_data((1, 100), num_instances)
        self.x2 = generate_data((101, 200), num_instances)
        self.x3 = generate_data((201, 300), num_instances)
        self.x4 = generate_data((301, 400), int(num_instances / 2))

    def _target_c(self, x):
        # ensuring that the function can handle
        # both 1D and 2D inputs uniformly.
        x = np.atleast_2d(x)

        # f(\mathbf{x}) = \sum_{i=1}^{n}
        # \theta_i \sqrt{\frac{\alpha}{N}} \cos
        # \left( \frac{\mathbf{w}_i \cdot
        # \mathbf{x}}{l} + b_i \right)
        return np.apply_along_axis(
            lambda x: np.sum(
                self.theta
                * np.sqrt(self.alpha / self.N)
                * np.cos(np.dot(self.w, x / self.l) + self.b)
            ),
            1,
            x,
        )

    def _target_s(self, x):
        return np.sin(x[:, 0]) + np.cos(x[:, 1])

    def __call__(self, clf, scenario):

        def classify_or_regress(data, func):
            return self._classify_output(func(data)) if clf else func(data)

        def _gen_df(x, y, task_num):
            df = pd.DataFrame(
                np.column_stack((x, y)), columns=["Feature 1", "Feature 2", "target"]
            )
            df["Task"] = np.ones_like(y) * task_num
            return df

        if scenario == 1:
            """Included the common task only.
            Data Pooling should be optimal.
            """
            # One random function with unique randomness values.
            # (calling the constructor one time).
            tg = FuncGen(self.n)
            y1 = classify_or_regress(tg.x1, self._target_c)
            y2 = classify_or_regress(tg.x2, self._target_c)
            y3 = classify_or_regress(tg.x3, self._target_c)
            y4 = classify_or_regress(tg.x4, self._target_c)
            ranges = [(tg.x1, y1), (tg.x2, y2), (tg.x3, y3), (tg.x4, y4)]

        elif scenario == 2:
            """Included the specific task only.
            Single-task learning should be optimal.
            """
            # Different functions per task without a common function.
            tg1 = FuncGen(self.n)
            tg2 = FuncGen(self.n)
            tg3 = FuncGen(self.n)
            tg4 = FuncGen(self.n)
            y1 = classify_or_regress(tg1.x1, self._target_s)
            y2 = classify_or_regress(tg2.x2, self._target_s)
            y3 = classify_or_regress(tg3.x3, self._target_s)
            y4 = classify_or_regress(tg1.x1, self._target_s)
            ranges = [(tg1.x1, y1), (tg2.x2, y2), (tg3.x3, y3), (tg1.x1, y4)]
        elif scenario == 3:
            """Multi-Task setting.
            Multi-Task learning should be optimal.
            """
            # Calling the constructor various times
            # to create diverse weights (self.b, self.w) for the common task.
            tg1 = FuncGen(self.n)
            tg2 = FuncGen(self.n)
            tg3 = FuncGen(self.n)
            tg4 = FuncGen(self.n)
            y1 = classify_or_regress(tg1.x1, self._target_c)
            y2 = classify_or_regress(tg2.x2, self._target_c)
            y3 = classify_or_regress(tg3.x3, self._target_c)
            y4 = classify_or_regress(tg4.x1, self._target_c)
            ranges = [(tg1.x1, y1), (tg2.x2, y2), (tg3.x3, y3), (tg4.x1, y4)]
        elif scenario == 4:
            """Multi-Task including an outlier,
            which is a task that doesn't have a common part.
            The robust Multi-task model should be optimal.
            """
            # Ideal objective: the Robust MT sould close to optimal in the rest of scenarios.
            # Reducing the specific task function by multiplying low w

            # _target_s should be different for each task.
            tg1 = FuncGen(self.n)
            tg2 = FuncGen(self.n)
            tg3 = FuncGen(self.n)
            tg4 = FuncGen(self.n)
            y1 = classify_or_regress(tg1.x1, lambda x: tg1._multi_task_gen(x))
            y2 = classify_or_regress(tg2.x2, lambda x: tg2._multi_task_gen(x))
            y3 = classify_or_regress(tg3.x3, lambda x: tg3._multi_task_gen(x))
            y4 = classify_or_regress(tg4.x4, tg4._target_s)
            ranges = [(tg1.x1, y1), (tg2.x2, y2), (tg3.x3, y3), (tg4.x4, y4)]

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))
        pd.concat(dfs, ignore_index=True).to_csv(
            f"{'clf' if clf else 'reg'}_{scenario}.csv"
        )

    def _multi_task_gen(self, x):
        return self._target_c(x) + 0.1 * self._target_s(x)

    def _classify_output(self, output):
        normalized_output = (output - np.mean(output)) / np.std(output)
        threshold = 0
        return np.where(normalized_output < threshold, 0, 1)
