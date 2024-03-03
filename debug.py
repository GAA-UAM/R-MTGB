# %%
from NoiseAwareGB.model import *
from Plots.plots import *
from Dataset.dataset import *
from Dataset.split import split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns

clf_data_gen = toy_clf_dataset(
    n_samples=2000,
    noise_sample=int(2000 * 1e-1),
    seed=111,
    noise_factor=0.1,
)

df = clf_data_gen("binary")
X, y, task = (
    df.drop(columns=["target", "task"]).values,
    df.target.values,
    df.task.values,
)
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

model_mt.fit(x_train, y_train, task_train)
pred_mt, pred_task = model_mt.predict(x_test, task_test)

# %%
thetas = model_mt.thetas

plt.plot(thetas[:, 0])
plt.plot(thetas[:, 1])
# %%

accuracy_score(y_test[task_test==0], pred_task[task_test==0])
# pred_task[task_test==0]
# task_test.shape