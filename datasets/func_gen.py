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

    def _gen_data(self, seed_range, num_samples):
        random_seed = np.random.randint(seed_range[0], seed_range[1])
        rng = np.random.default_rng(np.random.randint(random_seed))
        data = rng.uniform(-3, 3, size=(num_samples, self.d))
        return StandardScaler().fit_transform(data)

    def _gen_target(self):
        # ensuring that the function can handle
        # both 1D and 2D inputs uniformly.
        x = np.random.standard_normal(size=(self.n, self.d))
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

    def _multi_task_gen(self, c, t, w):
        return c() + (w * t())

    def _classify_output(self, output):
        threshold = np.mean(output)
        output = np.where(output > threshold, 0, 1).astype(int)
        return output

    def __call__(self, clf, scenario):

        def classify_or_regress(func):
            return self._classify_output(func()) if clf else func()

        def _gen_df(x, y, task_num):
            df = pd.DataFrame(
                np.column_stack((x, y)), columns=["Feature 1", "Feature 2", "target"]
            )
            df["Task"] = np.ones_like(y) * task_num
            return df

        def _input_space(tg, multi_task=False):

            ranges = [(1, 10), (11, 20), (21, 30), (31, 40)]
            ns = [self.n, self.n, self.n, self.n // 2 if multi_task else self.n]
            input_space = [tg._gen_data(ranges[i], ns[i]) for i in range(len(ranges))]
            return input_space

        def _output_space(multi_task=False, noise=False):
            ranges = [(41, 50), (51, 60), (61, 70), (71, 80)]

            if multi_task:
                # Diverse weights (self.b, self.w) for the specific task.
                task_generators = [FuncGen(self.n) for _ in range(4)]

                if not noise:
                    output_space = [
                        classify_or_regress(tg._gen_target) for tg in task_generators
                    ]
                elif noise:
                    # Adding weight to the last task.
                    output_space = [
                        (
                            classify_or_regress(tg._gen_target)
                            if i < 3
                            else classify_or_regress(
                                lambda: self._multi_task_gen(
                                    task_generators[0]._gen_target,
                                    tg._gen_target,
                                    0.1,
                                )
                            )
                        )
                        for i, tg in enumerate(task_generators)
                    ]
            else:
                # One random function with the unique seeds.F
                tg = FuncGen(self.n)
                output_space = [classify_or_regress(tg._gen_target) for _ in ranges]

            return output_space

        if scenario == 1:
            """Included the common task only.
            Data Pooling should be optimal.
            """
            y1, y2, y3, y4 = _output_space(multi_task=False, noise=False)
            x1, x2, x3, x4 = _input_space(FuncGen(self.n), False)

        elif scenario == 2:
            """Included the specific task only (different weights).
            Single-task learning should be optimal.
            """
            y1, y2, y3, y4 = _output_space(multi_task=True, noise=False)
            x1, x2, x3, x4 = _input_space(FuncGen(self.n), False)

        elif scenario == 3:
            """Multi-Task setting.
            Multi-Task learning should be optimal.
            """

            y1, y2, _, _ = _output_space(multi_task=False, noise=False)
            _, _, y3, y4 = _output_space(multi_task=True, noise=False)
            x1, x2, x3, x4 = _input_space(FuncGen(self.n), False)

        elif scenario == 4:
            """Robust Multitask learning.
            Multi-Task including an outlier (specific task weight=0.1),
            which is a task that doesn't have a common part.
            The robust Multi-task model should be optimal.
            """

            y1, y2, _, _ = _output_space(multi_task=False, noise=False)
            _, _, y3, y4 = _output_space(multi_task=True, noise=True)
            x1, x2, x3, x4 = _input_space(FuncGen(self.n), True)
            y4 = y4[: len(x4)]

        ranges = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))
        pd.concat(dfs, ignore_index=True).to_csv(
            f"{'clf' if clf else 'reg'}_{scenario}.csv"
        )
