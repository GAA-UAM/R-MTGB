# %%
from sklearn.model_selection import GridSearchCV
import pandas as pd
import json
import numpy as np
import warnings
import os
from scipy.special import expit as sigmoid
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from mtgb_models.mt_gb import MTGBClassifier
from mtgb_models.mt_gb import MTGBRegressor


warnings.simplefilter("ignore")


def data_pre_split(df):
    columns = [f"Feature {i}" for i in range(df.shape[1] - 2)]
    X = df[columns]
    y = df.Target
    Task = df.Task
    return X, y, Task


def extract_data(path, clf, train):
    clf_ = "clf_4" if clf else "reg_4"
    pre_title = "train_" if train else "test_"
    title = pre_title + clf_ + ".csv"
    df = pd.read_csv(os.path.join(path, title))
    df = df.iloc[:, 1:]

    return df


def to_csv(data, path, name):
    np.savetxt(rf"{path}/{name}.csv", data, delimiter=",")


class run:
    def __init__(
        self,
        max_depth,
        n_estimators,
        n_common_estimators,
        subsample,
        max_features,
        learning_rate,
        random_state,
        clf,
        path_exp,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_common_estimators = n_common_estimators

        problem = "clf" if clf else "reg"
        self.path = rf"{path_exp}/{problem}"

    def _mat(self, x_train, x_test, y_train, y_test):
        X_train = x_train.copy()
        X_test = x_test.copy()
        Y_train = y_train.copy()
        Y_test = y_test.copy()

        return X_train, X_test, Y_train, Y_test

    def hyperparameter_tuning(self, model, param_grid, score):
        """
        Perform hyperparameter tuning using GridSearchCV and return the best model.
        """

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring=score,
        )

        return grid_search

    def fit_clf(self, x_train, y_train, task_train, x_test, y_test, task_test):

        to_csv(np.column_stack((y_test, task_test)), self.path, f"y_test")
        to_csv(np.column_stack((y_train, task_train)), self.path, f"y_train")

        X_train, X_test, Y_train, Y_test = self._mat(x_train, x_test, y_train, y_test)

        def evaluate_model(
            param_grid,
            title,
            X_train,
            Y_train,
            task_train,
            X_test,
            Y_test,
            task_test,
        ):

            model_mt = MTGBClassifier(
                max_depth=self.max_depth,
                n_iter_1st=0,
                n_iter_2nd=50,
                n_iter_3rd=0,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
            )

            grid_search = self.hyperparameter_tuning(model_mt, param_grid, "accuracy")
            grid_search = grid_search.fit(
                np.column_stack((X_train, task_train)), Y_train, task_info=-1
            )

            best_params = grid_search.best_params_
            with open(os.path.join(self.path, f"best_params_{title}.json"), "w") as f:
                json.dump(best_params, f)
            model_mt = grid_search.best_estimator_
            model_mt.fit(np.column_stack((X_train, task_train)), Y_train, task_info=-1)
            pred_test_mt = model_mt.predict(np.column_stack((X_test, task_test)))
            pred_test_mt = np.column_stack((pred_test_mt, task_test))
            pred_train_mt = model_mt.predict(np.column_stack((X_train, task_train)))

            test_error = accuracy_score(
                y_test, model_mt.predict(np.column_stack((X_test, task_test)))
            )

            pred_train_mt = np.column_stack((pred_train_mt, task_train))
            train_error = accuracy_score(
                Y_train, model_mt.predict(np.column_stack((X_train, task_train)))
            )

            to_csv(pred_test_mt, self.path, f"pred_test_{title}")
            to_csv(pred_train_mt, self.path, f"pred_train_{title}")
            to_csv(sigmoid(model_mt.theta), self.path, f"sigmoid_theta_{title}")
            to_csv(train_error * np.ones((1, 1)), self.path, f"train_error_{title}")
            to_csv(test_error * np.ones((1, 1)), self.path, f"test_error_{title}")

        param_grid = {
            "n_iter_1st": [0, 20, 30, 50],
            "n_iter_2nd": [20, 30, 50],
            "n_iter_3rd": [0, 20, 30, 50],
        }

        evaluate_model(
            param_grid,
            "RMTB",
            X_train,
            Y_train,
            task_train,
            X_test,
            Y_test,
            task_test,
        )

        param_grid = {
            "n_iter_1st": [20, 30, 50],
            "n_iter_2nd": [0],
            "n_iter_3rd": [0, 20, 30, 50],
        }

        evaluate_model(
            param_grid,
            "MTB",
            X_train,
            Y_train,
            task_train,
            X_test,
            Y_test,
            task_test,
        )

        param_grid = {
            "n_iter_1st": [0],
            "n_iter_2nd": [0],
            "n_iter_3rd": [20, 30, 50],
        }

        evaluate_model(
            param_grid,
            "STL",
            X_train,
            Y_train,
            task_train,
            X_test,
            Y_test,
            task_test,
        )

        param_grid = {
            "n_iter_1st": [0],
            "n_iter_2nd": [0],
            "n_iter_3rd": [20, 30, 50],
        }

        evaluate_model(
            param_grid,
            "POOLING",
            X_train,
            Y_train,
            task_train * 0.0,
            X_test,
            Y_test,
            task_test * 0.0,
        )

        param_grid = {
            "n_iter_1st": [0],
            "n_iter_2nd": [0],
            "n_iter_3rd": [20, 30, 50],
        }

        X_train_poo_task_as_feature = np.column_stack(
            (
                X_train,
                np.eye(np.max(task_train.to_numpy().astype(int)) + 1)[
                    task_train.to_numpy().astype(int)
                ],
            )
        )
        X_test_poo_task_as_feature = np.column_stack(
            (
                X_test,
                np.eye(np.max(task_test.to_numpy().astype(int)) + 1)[
                    task_test.to_numpy().astype(int)
                ],
            )
        )

        evaluate_model(
            param_grid,
            "POOLING_TASK_AS_FEATURE",
            X_train_poo_task_as_feature,
            Y_train,
            task_train * 0.0,
            X_test_poo_task_as_feature,
            Y_test,
            task_test * 0.0,
        )

    def fit_reg(self, x_train, y_train, task_train, x_test, y_test, task_test):

        to_csv(np.column_stack((y_test, task_test)), self.path, f"y_test")
        to_csv(np.column_stack((y_train, task_train)), self.path, f"y_train")

        X_train, X_test, Y_train, Y_test = x_train, x_test, y_train, y_test

        def evaluate_model(
            param_grid, title, X_train, Y_train, task_train, X_test, Y_test, task_test
        ):

            # Proposed model training

            model_mt = MTGBRegressor(
                max_depth=self.max_depth,
                n_iter_1st=0,
                n_iter_2nd=50,
                n_iter_3rd=0,
                subsample=self.subsample,
                max_features=None,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
            )
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            grid_search = self.hyperparameter_tuning(model_mt, param_grid, scorer)

            grid_search = grid_search.fit(
                np.column_stack((X_train, task_train)), Y_train, task_info=-1
            )

            best_params = grid_search.best_params_
            with open(os.path.join(self.path, f"best_params_{title}.json"), "w") as f:
                json.dump(best_params, f)
            model_mt = grid_search.best_estimator_
            model_mt.fit(np.column_stack((X_train, task_train)), Y_train, task_info=-1)
            pred_test_mt = model_mt.predict(np.column_stack((X_test, task_test)))
            pred_test_mt = np.column_stack((pred_test_mt, task_test))
            pred_train_mt = model_mt.predict(np.column_stack((X_train, task_train)))
            test_error = np.mean(
                (model_mt.predict(np.column_stack((X_test, task_test))) - y_test) ** 2
            )

            pred_train_mt = np.column_stack((pred_train_mt, task_train))
            train_error = np.mean(
                (model_mt.predict(np.column_stack((X_train, task_train))) - y_train)
                ** 2
            )

            to_csv(pred_test_mt, self.path, f"pred_test_{title}")
            to_csv(pred_train_mt, self.path, f"pred_train_{title}")
            to_csv(sigmoid(model_mt.theta), self.path, f"sigmoid_theta_{title}")
            to_csv(train_error * np.ones((1, 1)), self.path, f"train_error_{title}")
            to_csv(test_error * np.ones((1, 1)), self.path, f"test_error_{title}")

        param_grid = {
            "n_iter_1st": [0, 20, 30, 50],
            "n_iter_2nd": [20, 30, 50],
            "n_iter_3rd": [0, 20, 30, 50],
        }

        evaluate_model(
            param_grid, "RMTB", X_train, Y_train, task_train, X_test, Y_test, task_test
        )

        param_grid = {
            "n_iter_1st": [20, 30, 50],
            "n_iter_2nd": [0],
            "n_iter_3rd": [0, 20, 30, 50],
        }

        evaluate_model(
            param_grid, "MTB", X_train, Y_train, task_train, X_test, Y_test, task_test
        )

        param_grid = {
            "n_iter_1st": [0],
            "n_iter_2nd": [0],
            "n_iter_3rd": [20, 30, 50],
        }

        evaluate_model(
            param_grid, "STL", X_train, Y_train, task_train, X_test, Y_test, task_test
        )

        param_grid = {
            "n_iter_1st": [0],
            "n_iter_2nd": [0],
            "n_iter_3rd": [20, 30, 50],
        }

        evaluate_model(
            param_grid,
            "POOLING",
            X_train,
            Y_train,
            task_train * 0.0,
            X_test,
            Y_test,
            task_test * 0.0,
        )

        param_grid = {
            "n_iter_1st": [0],
            "n_iter_2nd": [0],
            "n_iter_3rd": [20, 30, 50],
        }

        X_train_poo_task_as_feature = np.column_stack(
            (
                X_train,
                np.eye(np.max(task_train.to_numpy().astype(int)) + 1)[
                    task_train.to_numpy().astype(int)
                ],
            )
        )
        X_test_poo_task_as_feature = np.column_stack(
            (
                X_test,
                np.eye(np.max(task_test.to_numpy().astype(int)) + 1)[
                    task_test.to_numpy().astype(int)
                ],
            )
        )

        evaluate_model(
            param_grid,
            "POOLING_TASK_AS_FEATURE",
            X_train_poo_task_as_feature,
            Y_train,
            task_train * 0.0,
            X_test_poo_task_as_feature,
            Y_test,
            task_test * 0.0,
        )


if __name__ == "__main__":

    np.random.seed(0)
    experiment = r"D:\Ph.D\Programming\Py\NoiseAwareBoost\experiments\toy_data_comparison\10tasks_2outliers_5features_300training"
    for clf in [True, False]:
        for batch in range(1, 100 + 1):
            data_path = f"{experiment}/{batch}"
            run_exp = run(
                max_depth=1,
                n_estimators=100,
                n_common_estimators=80,
                subsample=1,
                max_features=None,
                learning_rate=1.0,
                random_state=111,
                clf=clf,
                path_exp=data_path,
            )

            path = os.path.abspath(data_path)

            df_train = extract_data(path, clf, True)
            df_test = extract_data(path, clf, False)

            x_train, y_train, task_train = data_pre_split(df_train)

            perm = np.random.permutation(x_train.shape[0])

            x_train = x_train.loc[perm]
            y_train = y_train.loc[perm]
            task_train = task_train.loc[perm]

            x_test, y_test, task_test = data_pre_split(df_test)

            if clf:
                run_exp.fit_clf(
                    x_train=x_train,
                    y_train=y_train,
                    task_train=task_train,
                    x_test=x_test,
                    y_test=y_test,
                    task_test=task_test,
                )
            else:
                run_exp.fit_reg(
                    x_train=x_train,
                    y_train=y_train,
                    task_train=task_train,
                    x_test=x_test,
                    y_test=y_test,
                    task_test=task_test,
                )
