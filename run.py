# %%
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import warnings
from mtgb_models.mt_gb import MTGBRegressor
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")


df = pd.read_csv("tasks_data.csv")

x_train, y_train, task_train = (
    df[["X1", "X2"]].values,
    df["Y"].values,
    df["Task"].values,
)
x_test, y_test, task_test = df[["X1", "X2"]].values, df["Y"].values, df["Task"].values


n_estimators = 300
n_common_estimators = 100
n_mid_estimators = 200
learning_rate = 1.0
model_mt = MTGBRegressor(
    max_depth=1,
    n_estimators=n_estimators,
    n_common_estimators=n_common_estimators,
    n_mid_estimators=n_mid_estimators,
    subsample=1.0,
    max_features=None,
    learning_rate=learning_rate,
    random_state=1,
    early_stopping=None,
)

model_mt.fit(np.column_stack((x_train, task_train)), y_train, task_info=-1)
pred_mt = model_mt.predict(np.column_stack((x_train, task_train)), task_info=-1)

# print(pred_mt[0:5])
# print(model_mt.theta)
# import pdb

# pdb.set_trace()

model_mt.theta