# %%

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from _test.plots import *
import pandas as pd
import warnings
import os

warnings.simplefilter("ignore")


def data_pre_split(df):
    X = df[["Feature 1", "Feature 2"]]
    y = df.target
    Task = df.Task
    return X, y, Task


class run:
    def __init__(
        self,
        max_depth,
        n_estimators,
        subsample,
        max_features,
        learning_rate,
        random_state,
        es,
        clf,
        path_exp,
        scenario,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.es = es
        self.scenario = scenario

        problem = "clf" if clf else "reg"
        self.path = rf"Results\{path_exp}\{problem}"

    def fit_clf(
        self, x_train, y_train, task_train, x_test, y_test, task_test, proposed_mtgb
    ):

        data_type = "classification"
        title = "Conventional MT" if not proposed_mtgb else "Proposed MT"

        if proposed_mtgb:
            from mtgb_models.mt_gb import MTGBClassifier

            model_mt = MTGBClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                early_stopping=self.es,
                step_size=0.1,
            )
            model_mt.fit(np.column_stack((x_train, task_train)), y_train, task_info=-1)
            pred_mt = model_mt.predict(
                np.column_stack((x_test, task_test)), task_info=-1
            )

            sigmoid_plot(model_mt, "Proposed MT", data_type, self.path)

        else:
            import sys

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
                n_common_estimators=20,
            )

            model_mt.fit(np.column_stack((x_train, task_train)), y_train, task_info=-1)
            pred_mt = model_mt.predict(np.column_stack((x_test, task_test)))

        np.savetxt(rf"{self.path}\pred_clf_{title}.csv", pred_mt, delimiter=",")

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

        model_st.fit(x_train, y_train)
        pred_st = model_st.predict(x_test)
        np.savetxt(rf"{self.path}\pred_GB_clf_.csv", pred_st, delimiter=",")

        training_score(model_mt, model_st, proposed_mtgb, data_type, self.path)

        if proposed_mtgb:
            np.savetxt(rf"{self.path}\pred_GB_DataPooling.csv", pred_st, delimiter=",")
        test_score_mt = np.zeros((model_mt.estimators_.shape[0]), dtype=np.float64)
        if proposed_mtgb:
            for i, y_pred in enumerate(
                model_mt.staged_predict(
                    np.column_stack((x_test, task_test)), task_info=-1
                )
            ):
                test_score_mt[i] = accuracy_score(y_test, y_pred)
        if not proposed_mtgb:
            for i, y_pred in enumerate(
                model_mt.staged_predict(np.column_stack((x_test, task_test)))
            ):
                test_score_mt[i] = accuracy_score(y_test, y_pred)
        test_score_st = np.zeros((model_st.estimators_.shape[0]), dtype=np.float64)
        for i, y_pred in enumerate(model_st.staged_predict(x_test)):
            test_score_st[i] = accuracy_score(y_test, y_pred)

        test_score_st_i = np.zeros(
            (self.n_estimators, len(set(task_train))), dtype=np.float64
        )
        pred_st_i = []
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
            model_st_i.fit(x_train[task_train == r], y_train[task_train == r])
            pred = model_st_i.predict(x_test[task_test == r])
            pred_st_i.append(pred)

            for i, y_pred in enumerate(model_st_i.staged_predict(x_test)):
                test_score_st_i[i, int(r)] = accuracy_score(y_test, y_pred)
        if proposed_mtgb:
            pd.DataFrame(pred_st_i).to_csv(rf"{self.path}\pred_GB_task_independent.csv")
        predict_stage_plot(
            test_score_mt,
            test_score_st,
            test_score_st_i,
            proposed_mtgb,
            data_type,
            self.path,
        )

        confusion_plot(
            data_type,
            proposed_mtgb,
            y_test,
            pred_st,
            pred_mt,
            task_test,
            pred_st_i,
            self.path,
        )

    def fit_reg(
        self, x_train, y_train, task_train, x_test, y_test, task_test, proposed_mtgb
    ):

        title = "Conventional MT" if not proposed_mtgb else "Proposed MT"
        data_type = "regression"
        if proposed_mtgb:
            from mtgb_models.mt_gb import MTGBRegressor

            model_mt = MTGBRegressor(
                max_depth=5,
                n_estimators=self.n_estimators,
                subsample=0.5,
                max_features="sqrt",
                learning_rate=0.05,
                random_state=1,
                criterion="squared_error",
                early_stopping=self.es,
                step_size=0.1,
            )

            model_mt.fit(np.column_stack((x_train, task_train)), y_train, task_info=-1)
            pred_mt = model_mt.predict(
                np.column_stack((x_test, task_test)), task_info=-1
            )
            mean_squared_error(pred_mt, y_test)
            sigmoid_plot(model_mt, title, data_type, self.path)

        else:
            import sys

            sys.path.append(r"D:\Ph.D\Programming\Py\MT-GB\MT_GB")
            from model.mtgb import MTGBRegressor

            model_mt = MTGBRegressor(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                n_iter_no_change=self.es,
                n_common_estimators=20,
            )

            model_mt.fit(np.column_stack((x_train, task_train)), y_train)
            pred_mt = model_mt.predict(np.column_stack((x_test, task_test)))

        np.savetxt(rf"{self.path}\pred_reg_{title}.csv", pred_mt, delimiter=",")

        model_st = GradientBoostingRegressor(
            max_depth=5,
            n_estimators=self.n_estimators,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
            n_iter_no_change=self.es,
        )

        model_st.fit(x_train, y_train)
        pred_st = model_st.predict(x_test)

        training_score(model_mt, model_st, proposed_mtgb, data_type, self.path)

        if proposed_mtgb:
            test_score_mt = np.zeros((model_mt.estimators_.shape[0]), dtype=np.float64)
            for i, y_pred in enumerate(
                model_mt.staged_predict(
                    np.column_stack((x_test, task_test)), task_info=-1
                )
            ):
                test_score_mt[i] = mean_squared_error(y_test, y_pred)
        else:
            test_score_mt = np.zeros((model_mt.estimators_.shape[0]), dtype=np.float64)
            for i, y_pred in enumerate(
                model_mt.staged_predict(np.column_stack((x_test, task_test)))
            ):
                test_score_mt[i] = mean_squared_error(y_test, y_pred)
        test_score_st = np.zeros((model_st.estimators_.shape[0]), dtype=np.float64)
        for i, y_pred in enumerate(model_st.staged_predict(x_test)):
            test_score_st[i] = mean_squared_error(y_test, y_pred)

        test_score_st_i = np.zeros(
            (self.n_estimators, len(set(task_train))), dtype=np.float64
        )
        np.savetxt(rf"{self.path}\pred_GB_reg_{title}.csv", pred_st, delimiter=",")
        pred_st_i = []
        for r in set(task_train):
            model_st_i = GradientBoostingRegressor(
                max_depth=5,
                n_estimators=self.n_estimators,
                subsample=0.5,
                max_features="sqrt",
                learning_rate=0.05,
                random_state=1,
                criterion="squared_error",
                n_iter_no_change=self.es,
            )
            model_st_i.fit(x_train[task_train == r], y_train[task_train == r])
            pred = model_st_i.predict(x_test[task_test == r])
            pred_st_i.append(pred)

            for i, y_pred in enumerate(model_st_i.staged_predict(x_test)):
                test_score_st_i[i, int(r)] = mean_squared_error(y_test, y_pred)

        if proposed_mtgb:
            pd.DataFrame(pred_st_i).to_csv(rf"{self.path}\pred_GB_task_independent.csv")
        predict_stage_plot(
            test_score_mt,
            test_score_st,
            test_score_st_i,
            proposed_mtgb,
            data_type,
            self.path,
        )
        reg_plot(
            data_type,
            proposed_mtgb,
            y_test,
            pred_st,
            pred_mt,
            task_test,
            pred_st_i,
            self.path,
        )


def extract_data(path, clf, scenario):

    clf_ = "clf_" if clf else "reg_"

    title = clf_ + f"{scenario}.csv"

    df = pd.read_csv(os.path.join(path, title))
    df = df.iloc[:, 1:]

    return df


if __name__ == "__main__":

    proposed_mtgb = False
    experiment = "24Jul"
    for clf in [True, False]:
        for scenario in [1, 2, 3, 4]:
            for path_exp in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
                path_exp = f"{experiment}\scenario_{scenario}\{path_exp}"
                run_exp = run(
                    max_depth=5,
                    n_estimators=100,
                    subsample=0.5,
                    max_features="sqrt",
                    learning_rate=1,
                    random_state=111,
                    es=None,
                    clf=clf,
                    path_exp=path_exp,
                    scenario=scenario,
                )

                train_path = (
                    rf"D:\Ph.D\Programming\Py\NoiseAwareBoost\Results\{path_exp}"
                )
                test_path = rf"D:\Ph.D\Programming\Py\NoiseAwareBoost\Results\{experiment}\scenario_{scenario}\test_data"

                df_train = extract_data(train_path, clf, scenario)
                df_test = extract_data(test_path, clf, scenario)

                x_train, y_train, task_train = data_pre_split(df_train)
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
