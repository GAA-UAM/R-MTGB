# %%
from model import *
from dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error

colors = ["r", "g", "b", "k", "y"]


def split(df, train_ratio, random_state):
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_size = int(train_ratio * len(df_shuffled))

    train_df = df_shuffled.iloc[:train_size, :]
    test_df = df_shuffled.iloc[train_size:, :]

    def _split(df):
        X, y, task = (
            df.drop(columns=["target", "task"]).values,
            df.target.values,
            df.task.values,
        )

        return (X, y, task)

    return _split(train_df), _split(test_df)


def plot(df, title, clf):
    fig, ax1 = plt.subplots(1, 2, figsize=(7, 3))
    if clf:
        for class_label in range(len(set(y))):
            ax1[0].scatter(
                df[(df["target"] == class_label) & (df["task"] == 0)].feature_0,
                df[(df["target"] == class_label) & (df["task"] == 0)].feature_1,
                color=colors[class_label],
            )

        for class_label in range(len(set(y))):
            ax1[1].scatter(
                df[(df["target"] == class_label) & (df["task"] == 1)].feature_0,
                df[(df["target"] == class_label) & (df["task"] == 1)].feature_1,
                color=colors[class_label],
                label=f"class label: {class_label}",
            )
    else:
        ax1[0].scatter(
            df[(df["task"] == 0)].feature_0,
            df[(df["task"] == 0)].target,
            color=colors[0],
        )

        ax1[1].scatter(
            df[(df["task"] == 1)].feature_0,
            df[(df["task"] == 1)].target,
            color=colors[0],
        )
    ax1[1].set_title("noised_data")
    ax1[0].set_title("original_data")
    if clf:
        fig.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f"Dataset: {title}")


if __name__ == "__main__":

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

    clf = False
    data_type = "circle"
    n_samples = 2000
    noise_sample = int(n_samples * 1e-2)
    noise_factor = 2
    
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
        plot(df, data_type, clf)
        train, test = split(df, 0.8, 111)
        x_train, y_train, task_train = train
        x_test, y_test, task_test = test

        model = Classifier(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )

        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        model.fit(x_train, y_train)
        pred_st = model.predict(x_test)

        model.fit(x_train, y_train, task_train)
        pred_mt = model.predict(x_test, task_test)

        sns.heatmap(confusion_matrix(y_test, pred_st), annot=True, ax=axs[0])
        sns.heatmap(confusion_matrix(y_test, pred_mt), annot=True, ax=axs[1])
        axs[0].set_title(f"Accuracy: {accuracy_score(y_test, pred_st)* 100:.3f}% - ST")
        axs[1].set_title(f"Accuracy: {accuracy_score(y_test, pred_mt)* 100:.3f}% - MT")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(f"Dataset: {data_type}")

    else:

        data_type = "sinusoidal"
        n_samples = 2000

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
        plot(df, data_type, clf)
        train, test = split(df, 0.8, 111)
        x_train, y_train, task_train = train
        x_test, y_test, task_test = test
        model = Regressor(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )

        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        model.fit(x_train, y_train)
        pred_st = model.predict(x_test)

        model.fit(x_train, y_train, task_train)
        pred_mt = model.predict(x_test, task_test)
        mean_squared_error(pred_mt, y_test)

        threshold = 0.0
        colors = np.where(y_test >= threshold, "blue", "red")
        axs[0].scatter(y_test, pred_st)
        axs[1].scatter(y_test, pred_mt)
        axs[0].set_title(f"RMSE: {mean_squared_error(pred_st, y_test):.2f} - ST")
        axs[1].set_title(f"RMSE: {mean_squared_error(pred_mt, y_test):.2f} - MT")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(f"Dataset: {data_type}")
