import numpy as np
import pandas as pd


class FuncGen:
    def __init__(
        self,
        num_dims,
        num_random_features=500,
        alpha=1.0,
        length_scale=0.25,
    ):

        self.N = num_random_features
        self.d = num_dims
        self.w = np.random.randn(self.N, self.d)
        self.b = np.random.uniform(0, 2 * np.pi, self.N)
        self.theta = np.random.randn(self.N)
        self.alpha = alpha
        self.l = length_scale * num_dims  # Smoother functions in higher dimensions

    def evaluate_function(self, x):
        output = np.apply_along_axis(
            lambda x: np.sum(
                self.theta
                * np.sqrt(2.0 * self.alpha / self.N)
                * np.cos(np.dot(self.w, x / self.l) + self.b)
            ),
            1,
            x,
        )
        return output


class GenerateDataset:

    def __init__(self, scenario):
        self.scenario = scenario

        self.common_funcgen = None
        self.random_states = np.random.choice(range(1, 10000), (1000,), replace=False)

    def _valid_class_prop(self, y, alpha):
        unique, counts = np.unique(y, return_counts=True)
        normalized_counts = counts / counts.sum()
        return len(unique) == 2 and all(normalized_counts >= alpha)

    def _classify_output(self, y):
        y = np.sign(y)
        # y = (y + 1) // 2
        return y

    # Here pooling is optimal
    def _gen_data_scenario_1(
        self, num_dims, num_tasks, num_instances, num_outlier_tasks
    ):

        valid = False
        while not valid:

            X, Y = [], []
            funcgen = FuncGen(num_dims=num_dims)

            for _ in range(num_tasks):
                x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
                y = funcgen.evaluate_function(x)

                if self.regression is False:
                    y = self._classify_output(y)
                    if not self._valid_class_prop(y, 0.1):
                        valid = False
                        break
                    else:
                        X.append(x)
                        Y.append(y)
                else:
                    X.append(x)
                    Y.append(y)

            if len(X) == num_tasks and len(Y) == num_tasks:
                valid = True
        return X, Y

    # Here single task learning is optimal
    def _gen_data_scenario_2(
        self, num_dims, num_tasks, num_instances, num_outlier_tasks
    ):

        X = list()
        Y = list()

        for _ in range(num_tasks):
            funcgen = FuncGen(num_dims=num_dims)

            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
            y = funcgen.evaluate_function(x)

            if self.regression is False:
                y = self._classify_output(y)

            if not self.regression and not self._valid_class_prop(y, 0.1):

                while not self._valid_class_prop(y, 0.1):
                    funcgen = FuncGen(num_dims=num_dims)

                    x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
                    y = funcgen.evaluate_function(x)

                    if self.regression is False:
                        y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    # Here multi-task learning is optimal
    def _gen_data_scenario_3(
        self, num_dims, num_tasks, num_instances, num_outlier_tasks=None
    ):

        valid = False
        while not valid:
            X, Y = [], []

            if self.common_funcgen is None:
                self.common_funcgen = FuncGen(num_dims=num_dims)
                self.specific_funcgens = [
                    FuncGen(num_dims=num_dims) for i in range(num_tasks)
                ]

            for i in range(num_tasks):

                x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0

                y = self.common_funcgen.evaluate_function(
                    x
                ) * 0.9 + 0.1 * self.specific_funcgens[i].evaluate_function(x)

                if self.regression is False:
                    y = self._classify_output(y)
                    if not self._valid_class_prop(y, 0.1):
                        valid = False
                        break
                    else:
                        X.append(x)
                        Y.append(y)
                else:
                    X.append(x)
                    Y.append(y)
            if len(X) == num_tasks and len(Y) == num_tasks:
                valid = True
        return X, Y

    # Here robust multi-task learning is optimal since
    # there are outlier tasks
    def _gen_data_scenario_4(
        self,
        num_dims,
        num_tasks,
        num_instances,
        num_outlier_tasks,
    ):
        """
        Generate synthetic data for scenario 4 with a mix
        of nonoutlier and outlier tasks.

        Parameters:
            num_dims (int): Number of dimensions/features.
            num_tasks (int): Total number of tasks.
            num_instances (int): Number of data instances per task.
            num_outlier_tasks (int): Number of outlier tasks.

        Returns:
            X (list of np.ndarray): Input features for all tasks.
            Y (list of np.ndarray): Output targets for all tasks.
        """

        common_weight = 0.9
        specific_weight = 1 - common_weight

        while True:
            X, Y = [], []
            # Initialize common function generator 

            self.common_funcgen = FuncGen(num_dims=num_dims)

            # Generate data for regular (non-outlier) tasks
            for _ in range(num_tasks - num_outlier_tasks):
                funcgen_specific = FuncGen(num_dims)
                x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
                y = (self.common_funcgen.evaluate_function(x) * common_weight) + (
                    specific_weight * funcgen_specific.evaluate_function(x)
                )
                if not self.regression:
                    y = self._classify_output(y)
                    if not self._valid_class_prop(y=y, alpha=0.1):
                        break  # imbalanced classification, restart entire generation
                X.append(x)
                Y.append(y)

            # Only proceed if all regular tasks were successfully generated
            if len(X) != num_tasks - num_outlier_tasks:
                continue

            # Generate data for outlier tasks using a different common function
            common_funcgen = FuncGen(num_dims=num_dims)
            for _ in range(num_outlier_tasks):
                x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
                funcgen_specific = FuncGen(num_dims)
                y = (common_funcgen.evaluate_function(x) * common_weight) + (
                    specific_weight * funcgen_specific.evaluate_function(x)
                )
                if not self.regression:
                    y = self._classify_output(y)
                    if not self._valid_class_prop(y, 0.1):
                        break  # imbalanced classification, restart entire generation
                X.append(x)
                Y.append(y)
            # Confirm all tasks were generated successfully
            if len(X) == num_tasks and len(Y) == num_tasks:
                return X, Y

    def __call__(
        self, regression, num_dims, num_tasks, num_instances, num_outlier_tasks
    ):

        self.regression = regression

        scenario_methods = {
            1: self._gen_data_scenario_1,
            2: self._gen_data_scenario_2,
            3: self._gen_data_scenario_3,
            4: self._gen_data_scenario_4,
        }

        def _gen_df(x, y, task_num):
            columns = [f"Feature {i}" for i in range(x.shape[1])]
            columns.append("Target")
            df = pd.DataFrame(
                np.column_stack((x, y)),
                columns=columns,
            )
            df["Task"] = np.ones_like(y) * task_num
            return df

        if self.scenario in scenario_methods:
            x_list, y_list = scenario_methods[self.scenario](
                num_dims, num_tasks, num_instances, num_outlier_tasks
            )

        ranges = []
        for x, y in zip(x_list, y_list):
            ranges.append((x, y))

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))

        return pd.concat(dfs, ignore_index=True)
