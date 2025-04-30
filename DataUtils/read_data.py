import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_csv_safely(path):
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def _preprocess_adult_data(df, task_column):

    unique_values = sorted(df[task_column].unique())
    value_map = {value: index for index, value in enumerate(unique_values)}
    df["Task"] = df[task_column].map(value_map)

    columns_to_drop = ["sex", "race"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)

    return df


def ReadData(dataset, random_state):

    if "adult" in dataset:
        if dataset == "adult_gender":
            task_col = "sex"
        elif dataset == "adult_race":
            task_col = "race"

        sub_dataset = dataset
        dataset = "adult"
        base_path = os.path.join(os.path.dirname(__file__), "..", "Datasets", dataset)
        data_train = read_csv_safely(
            os.path.join(base_path, f"{dataset}_train_data.csv")
        )
        data_test = read_csv_safely(os.path.join(base_path, f"{dataset}_test_data.csv"))

        data_train = _preprocess_adult_data(data_train, task_col)
        data_test = _preprocess_adult_data(data_test, task_col)

        target_train = read_csv_safely(
            os.path.join(base_path, f"{dataset}_train_target.csv")
        )
        target_test = read_csv_safely(
            os.path.join(base_path, f"{dataset}_test_target.csv")
        )
    else:
        base_path = os.path.join(os.path.dirname(__file__), "..", "Datasets", dataset)

        data = read_csv_safely(os.path.join(base_path, f"{dataset}_data.csv"))
        target = read_csv_safely(os.path.join(base_path, f"{dataset}_target.csv"))
        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=0.2, random_state=random_state
        )
    return (
        data_train.values,
        target_train.values.flatten(),
        data_test.values,
        target_test.values.flatten(),
    )


def plot(X, y, feature, approach):

    tasks = X[:, -1]
    features = X[:, :-1]
    num_features = features.shape[1]
    feature_names = [f"Feature_{i}" for i in range(num_features)]

    train_df = pd.DataFrame(features, columns=feature_names)
    train_df["Task"] = tasks

    task_feature_means = train_df.groupby("Task")[feature_names].mean()

    if approach == "mean":
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=task_feature_means, x=feature_names[0], y=feature_names[1])

        for task_id in task_feature_means.index:
            plt.text(
                task_feature_means.loc[task_id, feature_names[feature]],
                task_feature_means.loc[task_id, feature_names[feature + 1]],
                str(int(task_id)),
            )
        plt.title(f"Mean of {feature_names[0]} vs. Mean of {feature_names[1]} per Task")
        plt.xlabel(f"Mean {feature_names[0]}")
        plt.ylabel(f"Mean {feature_names[1]}")
        plt.grid(True)
        plt.show()
    else:
        plot_df = plot_df = pd.DataFrame(
            {
                "Selected_Feature": X[:, feature],
                "Target": y,
                "Task": X[:, -1],
            }
        )

        sns.scatterplot(
            data=plot_df,
            x="Selected_Feature",
            y="Target",
            hue="Task",
            palette="viridis", 
            alpha=0.7,
            s=20, 
        )
