# %%
from Model.model import *
from Plots.plots import *
from Dataset.dataset import *
from Dataset.split import split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns

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


def run(clf, data_type, n_samples, noise_prec, noise_factor, td):

    noise_sample = int(n_samples * noise_prec)
    if clf:
        clf_data_gen = toy_clf_dataset(
            n_samples=n_samples,
            noise_sample=noise_sample,
            seed=111,
            noise_factor=noise_factor,
        )

        df = clf_data_gen(data_type)
        X, y, task = (
            df.drop(columns=["target", "task"]).values,
            df.target.values,
            df.task.values,
        )
        scatter(df, data_type, clf)
        train, test = split(df, 0.8, 111)
        x_train, y_train, task_train = train
        x_test, y_test, task_test = test

        model_mt = Classifier(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )

        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        model_mt.fit(x_train, y_train, task_train)
        pred_mt = model_mt.predict(x_test, task_test)

        model_st = Classifier(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )

        model_st.fit(x_train, y_train)
        pred_st = model_st.predict(x_test)

        sns.heatmap(confusion_matrix(y_test, pred_st), annot=True, ax=axs[0])
        sns.heatmap(confusion_matrix(y_test, pred_mt), annot=True, ax=axs[1])
        axs[0].set_title(f"Accuracy: {accuracy_score(y_test, pred_st)* 100:.3f}% - ST")
        axs[1].set_title(f"Accuracy: {accuracy_score(y_test, pred_mt)* 100:.3f}% - MT")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(f"Dataset: {data_type}")

        fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor="w", edgecolor="k")
        axs = axs.ravel()
        boundaries(X, y, model_st, axs=axs[0], title="ST")
        boundaries(X, y, model_mt, axs=axs[1], title="MT")

    else:

        noise_sample = int(n_samples * 1e-1)
        reg_data_gen = toy_reg_dataset(
            n_samples=n_samples,
            noise_sample=noise_sample,
            seed=111,
            noise_factor=noise_factor,
        )

        df = reg_data_gen(data_type)
        X, y, task = (
            df.drop(columns=["target", "task"]).values,
            df.target.values,
            df.task.values,
        )
        scatter(df, data_type, clf)
        train, test = split(df, 0.8, 111)
        x_train, y_train, task_train = train
        x_test, y_test, task_test = test
        model_st = Regressor(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )

        fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=False, sharey=True)
        model_st.fit(x_train, y_train)
        pred_st = model_st.predict(x_test)

        model_mt = Regressor(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )

        model_mt.fit(x_train, y_train, task_train)
        pred_mt = model_mt.predict(x_test, task_test)
        mean_squared_error(pred_mt, y_test)

        threshold = 0.0
        axs[0].scatter(y_test, pred_st)
        axs[1].scatter(y_test, pred_mt)

        axs[0].set_xlabel("True Values")
        axs[0].set_ylabel("pred Values")
        axs[1].set_xlabel("True Values")

        axs[0].set_title(f"RMSE: {mean_squared_error(pred_st, y_test):.2f} - ST")
        axs[1].set_title(f"RMSE: {mean_squared_error(pred_mt, y_test):.2f} - MT")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(f"Dataset: {data_type}")

        if td:
            threeD_surface(model_mt, x_train, x_test, y_test, "MT")
            threeD_surface(model_st, x_train, x_test, y_test, "ST")
        else:
            surface(model_mt, x_train, x_test, y_test, "MT")
            surface(model_st, x_train, x_test, y_test, "ST")


if __name__ == "__main__":
    run(
        clf=False,
        data_type="linear",
        n_samples=500,
        noise_prec=1e-1,
        noise_factor=5,
        td=False,
    )
