import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TaskGenerator:
    def __init__(self, num_instances):

        self.N = 500  # Number of features, only to generate the output not the X size.
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
        self.alpha = 100  # Dataset size.
        self.l = 10.0

        def generate_data(seed_range, num_samples):
            random_seed = np.random.randint(*4)
            rng = np.random.default_rng(np.random.randint(random_seed))
            data = rng.uniform(-3, 3, size=(num_samples, self.D))
            return StandardScaler().fit_transform(data)

        self.x1 = generate_data((1, 100), num_instances)
        self.x2 = generate_data((101, 200), num_instances)
        self.x3 = generate_data((201, 300), num_instances)
        self.x4 = generate_data((301, 400), int(num_instances / 2))

    # TODO:
    # Seperate the code to generate the task and 
    # on the other hand generate the
    # input instances.

    # Including 4 tasks in each setting.

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
            # Included the common task only.
            # Data Pooling should be optimal.
            # One random function (calling the constructor one time).
            y1 = classify_or_regress(self.x1, self._target_c)
            y2 = classify_or_regress(self.x2, self._target_c)
            y3 = classify_or_regress(self.x3, self._target_c)
            ranges = [(self.x1, y1), (self.x2, y2), (self.x3, y3)]

        elif scenario == 2:
            # Included the specific task only.
            # The simple single-Task learning should be optimal.
            # One different function per task without common function.
            y4 = classify_or_regress(self.x4, self._target_s)
            ranges = [(self.x4, y4)]
        elif scenario == 3:
            # Multi-Task setting.
            # Multi-Task learning should be optimal.
            # TODO:
            # Chaning self.b and self.D only for this scenario
            # By creatining different obejcts of init.

            # Reducing the specific task function by multiplying low w

            # _target_s should be different for each task.

            y1 = classify_or_regress(self.x1, lambda x: self._multi_task_gen(x, 0.9))
            y2 = classify_or_regress(self.x2, lambda x: self._multi_task_gen(x, 0.9))
            y3 = classify_or_regress(self.x3, lambda x: self._multi_task_gen(x, 0.9))
            y4 = classify_or_regress(self.x4, lambda x: self._multi_task_gen(x, 0.1))
            y4 = classify_or_regress(self.x4, self._target_s)
            ranges = [(self.x1, y1), (self.x2, y2), (self.x3, y3), (self.x4, y4)]
        elif scenario == 4:
            # Including the outliers.
            # which is a task that it doesn't have a common part.
            # The robust Multi-task model should be optimal.
            # Ideal objective: the Robust MT sould close to optimal in the rest of scenarios.
            pass

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))
        pd.concat(dfs, ignore_index=True).to_csv(
            f"{'clf' if clf else 'reg'}_{scenario}.csv"
        )

    def _multi_task_gen(self, x, w):
        return self._target_c(x) + 0.1 * self._target_s(x)

    def _classify_output(self, output):
        normalized_output = (output - np.mean(output)) / np.std(output)
        threshold = 0
        return np.where(normalized_output < threshold, 0, 1)
