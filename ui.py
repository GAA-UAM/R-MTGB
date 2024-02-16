# %%
from model import GradientBoostingClassifier
from dataset import toy_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

df = toy_data(n_samples=2000, n_classes=3)

X, y, task = (
    df.drop(columns=["target", "task"]).values,
    df.target.values,
    df.task.values,
)

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


def plot(df):
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


# plot(df)
train, test = split(df, 0.8, 111)
x_train, y_train, task_train = train
x_test, y_test, task_test = test

model = GradientBoostingClassifier(n_estimators=200)
model.fit(x_train, y_train, task_train)#, task_train)
pred = model.predict(x_test, task_test) #, task_test

print(accuracy_score(y_test, pred))
sns.heatmap(confusion_matrix(y_test, pred), annot=True)
#%%