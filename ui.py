# %%
from model import GradientBoostingClassifier
from dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


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


def plot(df, title):
    fig, ax1 = plt.subplots(1, 2, figsize=(7, 3))
    for class_label in range(len(set(y))):
        ax1[0].scatter(
            df[(df["target"] == class_label) & (df["task"] == 0)].feature_0,
            df[(df["target"] == class_label) & (df["task"] == 0)].feature_1,
            color=colors[class_label],
            label=f"original_data_class_{class_label}",
        )

    for class_label in range(len(set(y))):
        ax1[1].scatter(
            df[(df["target"] == class_label) & (df["task"] == 1)].feature_0,
            df[(df["target"] == class_label) & (df["task"] == 1)].feature_1,
            color=colors[class_label],
            label=f"noised_data_class_{class_label}",
        )
    ax1[1].set_title("noised_data")
    ax1[0].set_title("original_data")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f"Dataset: {title}")


if __name__ == "__main__":

    # List of available datasets

    # ["binary",
    # "multi_class",
    # "overlapping",
    # "correlated_noise",
    # "imbalanced_data",
    # "circle"]

    data_type = "circle"
    n_samples = 2000
    noise_sample = int(n_samples * 1e-2)
    dataset_generator = toy_dataset(
        n_samples=n_samples, noise_sample=noise_sample, seed=111, noise_factor=10
    )

    df = dataset_generator(data_type)
    X, y, task = (
        df.drop(columns=["target", "task"]).values,
        df.target.values,
        df.task.values,
    )
    plot(df, data_type)
    train, test = split(df, 0.8, 111)
    x_train, y_train, task_train = train
    x_test, y_test, task_test = test

    model = GradientBoostingClassifier(
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
