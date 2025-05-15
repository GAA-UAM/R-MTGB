import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.base import clone
from scipy.special import expit as sigmoid
from sklearn.model_selection import GridSearchCV


current_file_path = Path(__file__).resolve()
script_dir = current_file_path.parent
project_root = script_dir.parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from DataUtils.read_data import *
from mtgb_models.mt_gb import MTGBClassifier
from mtgb_models.mt_gb import MTGBRegressor


class RunExperiments:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)
        np.random.seed(int(seed))

    def _gridsearch(self, model, param_grid, score):

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=0,
            scoring=score,
        )

        return grid_search

    def _eval_model(
        self,
        dataset,
        model,
        param_grid,
        title,
        X_train,
        Y_train,
        task_train,
        X_test,
        Y_test,
        task_test,
    ):

        if "Regressor" in model.__class__.__name__:
            score_func = "neg_mean_squared_error"
        else:
            score_func = "accuracy"

        grid_search = self._gridsearch(model, param_grid, score_func)
        grid_search = grid_search.fit(np.column_stack((X_train, task_train)), Y_train)

        best_params = grid_search.best_params_
        with open(
            os.path.join(os.getcwd(), f"{dataset}_best_params_{title}.json"), "w"
        ) as f:
            json.dump(best_params, f)
        optimized_model = grid_search.best_estimator_
        optimized_model.fit(
            np.column_stack((X_train, task_train)), Y_train, task_info=-1
        )
        pred_test_mt = optimized_model.predict(np.column_stack((X_test, task_test)))
        pred_test_mt = np.column_stack((pred_test_mt, task_test))
        pred_train_mt = optimized_model.predict(np.column_stack((X_train, task_train)))
        pred_train_mt = np.column_stack((pred_train_mt, task_train))

        self._to_csv(pred_test_mt, f"{dataset}_pred_test_{title}")
        self._to_csv(pred_train_mt, f"{dataset}_pred_train_{title}")
        self._to_csv(
            sigmoid(optimized_model.theta),
            f"{dataset}_sigmoid_theta_{title}",
        )

    def _to_csv(self, data, name):
        np.savetxt(f"{name}.csv", data, delimiter=",")

    def fit(
        self,
        dataset,
        problem,
        X_train,
        Y_train,
        task_train,
        X_test,
        Y_test,
        task_test,
    ):

        if problem == "classification":
            model = MTGBClassifier(
                max_depth=1,
                n_iter_1st=0,
                n_iter_2nd=50,
                n_iter_3rd=0,
                subsample=1,
                max_features=None,
                learning_rate=0.1,
                random_state=self.seed,
                criterion="squared_error",
            )
        else:
            model = MTGBRegressor(
                max_depth=1,
                n_iter_1st=0,
                n_iter_2nd=50,
                n_iter_3rd=0,
                subsample=1,
                max_features=None,
                learning_rate=0.1,
                random_state=self.seed,
                criterion="squared_error",
            )
        configs = {
            "RMTB": {
                "n_iter_1st": [0, 20, 30, 50],
                "n_iter_2nd": [20, 30, 50],
                "n_iter_3rd": [0, 20, 30, 50, 100],
            },
            "MTB": {
                "n_iter_1st": [20, 30, 50],
                "n_iter_2nd": [0],
                "n_iter_3rd": [0, 20, 30, 50, 100],
            },
            "STL": {
                "n_iter_1st": [0],
                "n_iter_2nd": [0],
                "n_iter_3rd": [20, 30, 50, 100],
            },
            "POOLING": {
                "n_iter_1st": [0],
                "n_iter_2nd": [0],
                "n_iter_3rd": [20, 30, 50, 100],
            },
            "POOLING_TASK_AS_FEATURE": {
                "n_iter_1st": [0],
                "n_iter_2nd": [0],
                "n_iter_3rd": [20, 30, 50, 100],
            },
        }

        self._to_csv(np.column_stack((Y_test, task_test)), f"{dataset}_y_test")
        self._to_csv(np.column_stack((Y_train, task_train)), f"{dataset}_y_train")

        for config_name, param_grid in configs.items():
            print(
                f"Running model {config_name} on dataset {dataset}...",
                flush=True,
                file=sys.stderr,
                end="\r",
            )
            if config_name == "POOLING":
                task_train_used = task_train * 0.0
                task_test_used = task_test * 0.0
                X_train_used = X_train
                X_test_used = X_test

            elif config_name == "POOLING_TASK_AS_FEATURE":
                X_train_used = np.column_stack(
                    (X_train, np.eye(int(task_train.max()) + 1)[task_train.astype(int)])
                )
                X_test_used = np.column_stack(
                    (X_test, np.eye(int(task_test.max()) + 1)[task_test.astype(int)])
                )
                task_train_used = task_train * 0.0
                task_test_used = task_test * 0.0

            else:
                X_train_used = X_train
                X_test_used = X_test
                task_train_used = task_train
                task_test_used = task_test

            self._eval_model(
                dataset,
                clone(model),
                param_grid,
                config_name,
                X_train_used,
                Y_train,
                task_train_used,
                X_test_used,
                Y_test,
                task_test_used,
            )


def run(dataset, seed):

    def split_task(X):
        task = X[:, -1]
        X_data = np.delete(X, -1, axis=1).astype(float)
        return X_data, task

    run_experiments = RunExperiments(dataset, seed)

    x_train, y_train, x_test, y_test = ReadData(dataset=dataset, random_state=int(seed))

    x_train, task_train = split_task(x_train)
    x_test, task_test = split_task(x_test)

    if dataset in ["school", "computer", "parkinson"]:
        problem = "regression"
    else:
        problem = "classification"

    run_experiments.fit(
        dataset,
        problem,
        x_train,
        y_train,
        task_train,
        x_test,
        y_test,
        task_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    run(args.dataset, args.seed)
