import numpy as np
import pandas as pd


class FuncGen:
    def __init__(
        self,
        num_dims,
        num_random_features=500,
        alpha=1.0,
        length_scale=0.125,
        random_state=111,
    ):

        self.N = num_random_features
        self.d = num_dims
        self.w = np.random.randn(self.N, self.d)
        self.b = np.random.uniform(0, 2 * np.pi, self.N)
        self.theta = np.random.randn(self.N)
        self.alpha = alpha
        self.l = length_scale * num_dims  # Smoother functions in higher dimensions

        np.random.seed(random_state)

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
        np.random.seed(np.random.choice(range(1, 100), 1, replace=False)[0])

    def _valid_class_prop(self, y, alpha):
        unique, counts = np.unique(y, return_counts=True)
        normalized_counts = counts / counts.sum()
        return len(unique) == 2 and all(normalized_counts >= alpha)

    def _classify_output(self, y):
        y = np.sign(y)
        y = (y + 1) // 2
        return y

    # Here pooling is optimal
    def _gen_data_scenario_1(self, num_dims=2, num_tasks=8, num_instances=100):

        valid = False
        while not valid:
            random_states = np.random.choice(range(1, 100), 1, replace=False)

            X, Y = [], []
            funcgen = FuncGen(num_dims=num_dims, random_state=random_states[0])

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
    def _gen_data_scenario_2(self, num_dims=2, num_tasks=8, num_instances=100):

        X = list()
        Y = list()

        for _ in range(num_tasks):
            random_states = np.random.choice(range(1, 100), 1, replace=False)
            funcgen = FuncGen(num_dims=num_dims, random_state=random_states[0])

            x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
            y = funcgen.evaluate_function(x)

            if self.regression is False:
                y = self._classify_output(y)

            if not self.regression and not self._valid_class_prop(y, 0.1):

                while not self._valid_class_prop(y, 0.1):
                    random_states = np.random.choice(range(1, 100), 1, replace=False)
                    funcgen = FuncGen(num_dims=num_dims, random_state=random_states[0])

                    x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
                    y = funcgen.evaluate_function(x)

                    if self.regression is False:
                        y = self._classify_output(y)

            X.append(x)
            Y.append(y)

        return X, Y

    # Here multi-task learning is optimal
    def _gen_data_scenario_3(self, num_dims=2, num_tasks=8, num_instances=100):

        valid = False
        while not valid:
            X, Y = [], []
            random_states = np.random.choice(range(1, 10000), 1, replace=False)
            common_funcgen = FuncGen(num_dims=num_dims, random_state=random_states[0])

            for _ in range(num_tasks):

                specific_funcgen = FuncGen(
                    num_dims=num_dims, random_state=random_states[0]
                )

                x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0

                y = common_funcgen.evaluate_function(
                    x
                ) * 0.9 + 0.1 * specific_funcgen.evaluate_function(x)

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
        self, num_dims, num_tasks, num_instances, num_outlier_tasks=1
    ):

        valid = False
        X, Y = [], []
        while not valid:
            random_states_common = np.random.choice(range(1, 10000), 1, replace=False)

            common_funcgen = FuncGen(
                num_dims=num_dims, random_state=random_states_common[0]
            )

            for i in range(num_tasks):

                random_states_task_specific = np.random.choice(
                    range(10000, 20000), 1, replace=False
                )

                specific_funcgen = FuncGen(
                    num_dims=num_dims, random_state=random_states_task_specific[0]
                )
                x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0

                if i < num_tasks - num_outlier_tasks:
                    # non-outlier tasks
                    common_weight = 0.9
                    specific_weight = 1 - common_weight
                    y = (common_funcgen.evaluate_function(x) * common_weight) + (
                        specific_weight * specific_funcgen.evaluate_function(x)
                    )
                else:
                    # outlier tasks
                    y = specific_funcgen.evaluate_function(x)

                if self.regression is False:
                    y = self._classify_output(y)
                    if not self._valid_class_prop(y, 0.03):
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

    def __call__(self, regression, num_dims=5, num_tasks=8, num_instances=125):

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
            x_list, y_list = scenario_methods[self.scenario](num_dims, num_tasks, num_instances)

        ranges = []
        for x, y in zip(x_list, y_list):
            ranges.append((x, y))

        dfs = []
        for i, (x, y) in enumerate(ranges):
            dfs.append(_gen_df(x, y, i))

        return pd.concat(dfs, ignore_index=True)
