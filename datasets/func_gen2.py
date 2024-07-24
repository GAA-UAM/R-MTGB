import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FuncGen:
    def __init__(
        self,
        num_dims,
        num_random_features=500,
        alpha=5.0,
        length_scale=10.0,
    ):

        self.N = num_random_features
        self.d = num_dims
        self.w = np.random.randn(self.N, self.d)
        self.b = np.random.uniform(0, 2 * np.pi, self.N)
        self.theta = np.random.randn(self.N)
        self.alpha = alpha
        self.l = length_scale * num_dims  # Smoother functions in higher dimensions

    def evaluate_function(self, x, regression):
        output = np.apply_along_axis(
            lambda x: np.sum(
                self.theta
                * np.sqrt(self.alpha / self.N)
                * np.cos(np.dot(self.w, x / self.l) + self.b)
            ),
            1,
            x,
        )

        if not regression:
            if np.all(output >= 0) or np.all(output <= 0):
                # To make sure of balanced classification
                bias = np.mean(output)
                output -= bias
        return output


class GenerateDataset:

    # We have a parameter to indicate regresion,
    # Otherwise it is binary
    # classification with labels -1 and 1

    def __init__(self, scenario):
        self.scenario = scenario

    def _classify_output(self, y):
        y = np.sign(y)
        y = (y + 1) // 2
        return y

    # Here pooling is optimal
    def _gen_data_scenario_1(self, num_dims=2, num_tasks=5, num_instances=100):

        X = list()
        Y = list()

        funcgen = FuncGen(num_dims)

        for _ in range(num_tasks):
            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
            y = funcgen.evaluate_function(x, self.regression)

            if self.regression is False:
                y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    # Here single task learning is optimal
    def _gen_data_scenario_2(self, num_dims=2, num_tasks=5, num_instances=100):

        X = list()
        Y = list()

        for _ in range(num_tasks):
            funcgen = FuncGen(num_dims)

            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
            y = funcgen.evaluate_function(x, self.regression)

            if self.regression is False:
                y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    # Here multi-task learning is optimal

    def _gen_data_scenario_3(self, num_dims=2, num_tasks=5, num_instances=100):

        X = list()
        Y = list()

        common_funcgen = FuncGen(num_dims)

        for _ in range(num_tasks):

            specific_funcgen = FuncGen(num_dims)

            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0

            y = common_funcgen.evaluate_function(
                x, self.regression
            ) * 0.9 + 0.1 * specific_funcgen.evaluate_function(
                x, self.regression
            )  # Higher weight to the common part

            if self.regression is False:
                y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    # Here robust multi-task learning is optimal since
    # there are outlier tasks
    def _gen_data_scenario_4(
        self, num_dims=2, num_tasks=5, num_instances=100, num_outlier_tasks=2
    ):

        X = list()
        Y = list()

        common_funcgen = FuncGen(num_dims)

        for i in range(num_tasks):

            specific_funcgen = FuncGen(num_dims)

            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0

            if i >= num_tasks - num_outlier_tasks:
                y = specific_funcgen.evaluate_function(
                    x, self.regression
                )  # No common part
            else:
                y = common_funcgen.evaluate_function(
                    x, self.regression
                ) * 0.9 + 0.1 * specific_funcgen.evaluate_function(
                    x, self.regression
                )  # Higher weight to the common part

            if self.regression is False:
                y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    def __call__(self, regression, test):

        self.regression = regression

        scenario_methods = {
            1: self._gen_data_scenario_1,
            2: self._gen_data_scenario_2,
            3: self._gen_data_scenario_3,
            4: self._gen_data_scenario_4,
        }

        def _gen_df(x, y, task_num):
            df = pd.DataFrame(
                np.column_stack((x, y)), columns=["Feature 1", "Feature 2", "target"]
            )
            df["Task"] = np.ones_like(y) * task_num
            return df

        num_instances = 100 if not test else 1000
        if self.scenario in scenario_methods:
            x_list, y_list = scenario_methods[self.scenario](
                num_instances=num_instances
            )

        ranges = []
        for x, y in zip(x_list, y_list):
            ranges.append((x, y))

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))
        pd.concat(dfs, ignore_index=True).to_csv(
            f"{'clf' if not self.regression else 'reg'}_{self.scenario}.csv"
        )
