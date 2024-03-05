# %%

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from Plots.plots import *
from Dataset.dataset import *
from Dataset.split import split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# List of regression datasets
# "linear",
# "sinusoidal",
# "full_circle"]

# List of classification datasets
# ["binary",
# "multi_class",
# "overlapping",
# "correlated_noise",
# "imbalanced_data",
# "circle"]


class run:
    def __init__(
        self,
        n_samples,
        mean,
        noise_prec,
        noise_factor,
        max_depth,
        n_estimators,
        subsample,
        max_features,
        learning_rate,
        random_state,
    ):
        self.n_samples = n_samples
        self.mean = mean
        self.noise_prec = noise_prec
        self.noise_factor = noise_factor
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.random_state = random_state

        if self.noise_prec:
            self.noise_sample = int(self.n_samples * self.noise_prec)
        else:
            self.noise_sample = False

    def data_pre_split(self, df):
        X, y, task = (
            df.drop(columns=["target", "task"]).values,
            df.target.values,
            df.task.values,
        )

        return X, y, task

    def fit_clf(self, noise_mt, data_type, es):

        clf_data_gen = toy_clf_dataset(
            n_samples=self.n_samples,
            noise_sample=self.noise_sample,
            seed=self.random_state,
            noise_factor=self.noise_factor,
            mean=self.mean,
        )

        df = clf_data_gen(data_type)
        X, y, task = self.data_pre_split(df)
        scatter(df, data_type, True)
        train, test = split(df, 0.8, 111)
        x_train, y_train, task_train = train
        x_test, y_test, task_test = test

        if noise_mt:
            from NoiseAwareGB.model import Classifier

            model_mt = Classifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_features=self.max_features,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                criterion="squared_error",
                early_stopping=es,
            )
            model_mt.fit(X=x_train, y=y_train, task=task_train)
            pred_mt = model_mt.predict(x_test, task_test)

            theta_plot(model_mt, "Proposed MT", data_type)

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
                n_iter_no_change=es,
            )

            model_mt.fit(np.column_stack((x_train, task_train)), y_train)
            pred_mt = model_mt.predict(np.column_stack((x_test, task_test)))

        fig, ax = plt.subplots(1, 1)
        test_score = np.zeros((model_mt.estimators_.shape[0]), dtype=np.float64)
        for i, y_pred in enumerate(model_mt.staged_predict(x_test, task_test)):
            test_score[i] = accuracy_score(y_test, y_pred)
        ax.plot(test_score)

        model_st = GradientBoostingClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            max_features=self.max_features,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            criterion="squared_error",
            n_iter_no_change=es,
        )

        model_st.fit(x_train, y_train, True)
        pred_st = model_st.predict(x_test)

        training_score(model_mt, model_st, noise_mt, data_type)

        preds_st = []
        for r in set(task):
            model_st.fit(x_train[task_train == r], y_train[task_train == r], True)
            pred = model_st.predict(x_test[task_test == r])
            preds_st.append(pred)

        confusion_plot(
            data_type,
            noise_mt,
            y_test,
            pred_st,
            pred_mt,
            task_test,
            preds_st,
        )

        _, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor="w", edgecolor="k")
        axs = axs.ravel()
        boundaries(X, y, model_st, axs=axs[0], title="GB (Data Pooling)")
        if noise_mt:
            boundaries(X, y, model_mt, axs=axs[1], title=f"Proposed MT")

    def fit_reg(self, noise_mt, data_type, es):

        reg_data_gen = toy_reg_dataset(
            n_samples=self.n_samples,
            noise_sample=self.noise_sample,
            seed=self.random_state,
            noise_factor=self.noise_factor,
            mean=self.mean,
        )

        df = reg_data_gen(data_type)
        scatter(df, data_type, False)
        X, y, task = self.data_pre_split(df)
        train, test = split(df, 0.8, 111)
        x_train, y_train, task_train = train
        x_test, y_test, task_test = test

        title = "Conventional MT" if not noise_mt else "Proposed MT"

        if noise_mt:
            from NoiseAwareGB.model import Regressor

            model_mt = Regressor(
                max_depth=5,
                n_estimators=100,
                subsample=0.5,
                max_features="sqrt",
                learning_rate=0.05,
                random_state=1,
                criterion="squared_error",
                early_stopping=es,
            )

            model_mt.fit(x_train, y_train, task_train)
            pred_mt = model_mt.predict(x_test, task_test)
            mean_squared_error(pred_mt, y_test)
            theta_plot(model_mt, title, data_type)

            fig, ax = plt.subplots(1, 1)
            test_score = np.zeros((model_mt.estimators_.shape[0]), dtype=np.float64)
            for i, y_pred in enumerate(model_mt.staged_predict(x_test, task_test)):
                test_score[i] = mean_squared_error(y_test, y_pred)
            ax.plot(test_score)

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
                n_iter_no_change=es,
            )

            model_mt.fit(np.column_stack((x_train, task_train)), y_train)
            pred_mt = model_mt.predict(np.column_stack((x_test, task_test)))

        model_st = GradientBoostingRegressor(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
            n_iter_no_change=es,
        )

        model_st.fit(x_train, y_train)
        pred_st = model_st.predict(x_test)

        training_score(model_mt, model_st, title, data_type)

        preds_st = []
        for r in set(task):
            model_st.fit(x_train[task_train == r], y_train[task_train == r])
            pred = model_st.predict(x_test[task_test == r])
            preds_st.append(pred)

        
        reg_plot(
            data_type,
            noise_mt,
            y_test,
            pred_st,
            pred_mt,
            task_test,
            preds_st,
        )


if __name__ == "__main__":
    run_exp = run(
        n_samples=1000,
        mean=0,
        noise_prec=1e-1,
        noise_factor=5,
        max_depth=5,
        n_estimators=100,
        subsample=0.5,
        max_features="sqrt",
        learning_rate=5e-1,
        random_state=111,
    )

    # run_exp.fit_clf(noise_mt=True, data_type="circle", es=3)
    # run_exp.fit_reg(noise_mt=True, data_type="linear", es=3)
