import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_safely(path):
    df = pd.read_csv(path)
    return df


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

    if target_train.ndim > 1:
        target_train = target_train.values.ravel()
        target_test = target_test.values.ravel()
    else:
        target_train = target_train.values
        target_test = target_test.values

    return (
        data_train.values,
        target_train,
        data_test.values,
        target_test,
    )
