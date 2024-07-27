import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FuncGen:
    def __init__(
        self,
        num_dims,
        num_random_features=500,
        alpha=5.0,
        length_scale=3,
    ):

        self.N = num_random_features
        self.d = num_dims
        self.w = np.random.randn(self.N, self.d)
        self.b = np.random.uniform(0, 2 * np.pi, self.N)
        self.theta = np.random.randn(self.N)
        self.alpha = alpha
        self.l = length_scale * num_dims  # Smoother functions in higher dimensions

        # Generate a 2D grid of points
        x1 = np.linspace(-5, 5, 100)
        x2 = np.linspace(-5, 5, 100)
        x1, x2 = np.meshgrid(x1, x2)
        points = np.c_[x1.ravel(), x2.ravel()]

        # z = self.evaluate_function(points).reshape(x1.shape)
        # plt.figure(figsize=(10, 8))
        # plt.contourf(x1, x2, z, levels=50, cmap="viridis")
        # plt.colorbar(label="Function Value")
        # plt.title("Contour plot of the random function")
        # plt.xlabel("x1")
        # plt.ylabel("x2")
        # plt.show()

    def evaluate_function(self, x):
        output = np.apply_along_axis(
            lambda x: np.sum(
                self.theta
                * np.sqrt(self.alpha / self.N)
                * np.cos(np.dot(self.w, x / self.l) + self.b)
            ),
            1,
            x,
        )

        return output


class GenerateDataset:

    # We have a parameter to indicate regresion,
    # Otherwise it is binary
    # classification with labels -1 and 1

    def __init__(self, scenario):
        self.scenario = scenario

    def _classify_output(self, y):
        y = np.sign(y)
        # y = (y + 1) // 2
        return y

    # Here pooling is optimal
    def _gen_data_scenario_1(self, num_dims=2, num_tasks=5, num_instances=100):

        X = list()
        Y = list()

        funcgen = FuncGen(num_dims)

        for _ in range(num_tasks):
            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
            y = funcgen.evaluate_function(x)

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
            y = funcgen.evaluate_function(x)

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
                x
            ) * 0.9 + 0.1 * specific_funcgen.evaluate_function(
                x
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
                y = specific_funcgen.evaluate_function(x)  # No common part
            else:
                y = common_funcgen.evaluate_function(
                    x
                ) * 0.9 + 0.1 * specific_funcgen.evaluate_function(
                    x
                )  # Higher weight to the common part

            if self.regression is False:
                y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    def __call__(self, regression):

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

        if self.scenario in scenario_methods:
            x_list, y_list = scenario_methods[self.scenario](num_instances=2000)

        ranges = []
        for x, y in zip(x_list, y_list):
            ranges.append((x, y))

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))

        return pd.concat(dfs, ignore_index=True)
