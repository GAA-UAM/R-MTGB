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
