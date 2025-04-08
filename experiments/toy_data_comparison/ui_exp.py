# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import pandas as pd
import numpy as np
import warnings
import os
from scipy.special import expit as sigmoid
from sklearn.metrics import make_scorer, mean_squared_error

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
        es,
        clf,
        path_exp,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.es = es
        self.n_common_estimators = n_common_estimators

        problem = "clf" if clf else "reg"
        self.path = rf"{path_exp}/{problem}"

    def _mat(self, x_train, x_test, y_train, y_test):
        X_train = x_train.copy()
        X_test = x_test.copy()
        Y_train = y_train.copy()
        Y_test = y_test.copy()

        return X_train, X_test, Y_train, Y_test

    def hyperparameter_tuning(self, model, param_grid):
        """
        Perform hyperparameter tuning using GridSearchCV and return the best model.
        """
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring=scorer,
        )

        return grid_search

    def fit_clf(
        self, x_train, y_train, task_train, x_test, y_test, task_test, proposed_mtgb
    ):

        to_csv(np.column_stack((y_test, task_test)), self.path, f"y_test")
        to_csv(np.column_stack((y_train, task_test)), self.path, f"y_train")
        title = "Conventional MT" if not proposed_mtgb else "Proposed MT"

        if proposed_mtgb:
            import sys

            sys.path.append(r"../..")
            from mtgb_models.mt_gb import MTGBClassifier

            X_train, X_test, Y_train, Y_test = self._mat(
                x_train, x_test, y_train, y_test
            )

            model_mt = MTGBClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                early_stopping=self.es,
                n_common_estimators=self.n_common_estimators,
                verbose=0,
            )
            model_mt.fit(np.column_stack((X_train, task_train)), Y_train, task_info=-1)
            pred_test_mt = model_mt.predict(
                np.column_stack((X_test, task_test)), task_info=-1
            )
            pred_train_mt = model_mt.predict(
                np.column_stack((X_train, task_train)), task_info=-1
            )
            to_csv(pred_test_mt, self.path, f"pred_test_{title}")
            to_csv(pred_train_mt, self.path, f"pred_train_{title}")
            # to_csv(model_mt.sigmoid_thetas_[:, :], self.path, "sigmoid_theta")

            # Standard GB Data Pooling without task as the feature training
            X_train, X_test, Y_train, Y_test = self._mat(
                x_train, x_test, y_train, y_test
            )
            model_st = GradientBoostingClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                n_iter_no_change=self.es,
            )

            model_st.fit(X_train, Y_train)
            pred_test_st = model_st.predict(X_test)
            pred_train_st = model_st.predict(X_train)
            pred_test_st = np.column_stack((pred_test_st, task_test))
            pred_train_st = np.column_stack((pred_train_st, task_train))
            to_csv(pred_test_st, self.path, "pred_test_GB_datapooling")
            to_csv(pred_train_st, self.path, "pred_train_GB_datapooling")

            # Standard GB Data Pooling with task as the feature training
            X_train, X_test, Y_train, Y_test = self._mat(
                x_train, x_test, y_train, y_test
            )
            X_train = np.column_stack((X_train, task_train))
            X_test = np.column_stack((X_test, task_test))
            model_st = GradientBoostingClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                n_iter_no_change=self.es,
            )
            model_st.fit(X_train, Y_train)
            pred_test_st = model_st.predict(X_test)
            pred_train_st = model_st.predict(X_train)
            pred_test_st = np.column_stack((pred_test_st, task_test))
            pred_train_st = np.column_stack((pred_train_st, task_train))
            to_csv(pred_test_st, self.path, "pred_test_GB_datapooling_task_as_feature")
            to_csv(
                pred_train_st, self.path, "pred_train_GB_datapooling_task_as_feature"
            )

            # Standard GB single task learning

            X_train, X_test, Y_train, Y_test = self._mat(
                x_train, x_test, y_train, y_test
            )
            pred_test_list = []
            pred_train_list = []
            for r in set(task_train):
                model_st_i = GradientBoostingClassifier(
                    max_depth=self.max_depth,
                    n_estimators=self.n_estimators,
                    subsample=self.subsample,
                    max_features=self.max_features,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    criterion="squared_error",
                    n_iter_no_change=self.es,
                )
                model_st_i.fit(X_train[task_train == r], Y_train[task_train == r])
                preds_test = model_st_i.predict(X_test[task_test == r])
                task_column_test = np.full_like(preds_test, r)
                pred_test_list.append(np.column_stack((preds_test, task_column_test)))

                preds_train = model_st_i.predict(X_train[task_train == r])
                task_column_train = np.full_like(preds_train, r)
                pred_train_list.append(
                    np.column_stack((preds_train, task_column_train))
                )
            to_csv(np.vstack(pred_test_list), self.path, f"pred_test_GB_single_task")
            to_csv(np.vstack(pred_train_list), self.path, f"pred_train_GB_single_task")

        else:
            import sys

            X_train, X_test, Y_train, Y_test = self._mat(
                x_train, x_test, y_train, y_test
            )
            sys.path.append(r"D:\Ph.D\Programming\Py\MT-GB\MT_GB")
            from model.mtgb import MTGBClassifier

            model_mt = MTGBClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                n_iter_no_change=self.es,
                n_common_estimators=self.n_common_estimators,
            )

            model_mt.fit(np.column_stack((X_train, task_train)), Y_train, task_info=-1)
            pred_test_mt = model_mt.predict(np.column_stack((X_test, task_test)))
            pred_test_mt = np.column_stack((pred_test_mt, task_test))
            to_csv(pred_test_mt, self.path, f"pred_test_{title}")
            pred_train_mt = model_mt.predict(np.column_stack((X_train, task_train)))
            pred_train_mt = np.column_stack((pred_train_mt, task_train))
            to_csv(pred_train_mt, self.path, f"pred_train_{title}")

    def fit_reg(
        self, x_train, y_train, task_train, x_test, y_test, task_test, proposed_mtgb
    ):

        to_csv(np.column_stack((y_test, task_test)), self.path, f"y_test")
        to_csv(np.column_stack((y_train, task_train)), self.path, f"y_train")

        import sys

        sys.path.append(r"../../")
        from mtgb_models.mt_gb import MTGBRegressor

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

            grid_search = self.hyperparameter_tuning(
                model_mt,
                param_grid,
            )

            grid_search = grid_search.fit(
                np.column_stack((X_train, task_train)), Y_train, task_info=-1
            )

            print(f"Best parameters found: {grid_search.best_params_}")
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
            to_csv(sigmoid(model_mt.theta), self.path, "sigmoid_theta_{title}")
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
    proposed_mtgb = True
    experiment = "10tasks_2outliers_5features_300training"
    for clf in [False]:
        for batch in range(1, 100 + 1):
            print(batch)
            data_path = f"../../datasets/{experiment}/{batch}"
            run_exp = run(
                max_depth=1,
                n_estimators=100,
                n_common_estimators=80,
                subsample=1,
                max_features=None,
                learning_rate=1.0,
                random_state=111,
                es=None,
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
                    proposed_mtgb=proposed_mtgb,
                )
            else:
                run_exp.fit_reg(
                    x_train=x_train,
                    y_train=y_train,
                    task_train=task_train,
                    x_test=x_test,
                    y_test=y_test,
                    task_test=task_test,
                    proposed_mtgb=proposed_mtgb,
                )

# %%
