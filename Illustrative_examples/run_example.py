# %%
from plot import *
from models import *
from data_entry import *
from sklearn.metrics import mean_squared_error, accuracy_score


def model_to_fit(model_name, problem, data_pooling, all_features=None):

    x, y, task = data()

    metric = mean_squared_error if problem == "regression" else accuracy_score
    params = {
        "max_depth": 1,
        "n_estimators": 100,
        "n_common_estimators": 50,
        "learning_rate": 1.0,
        "random_state": 111,
        "problem": problem,
        "model": model_name,
    }

    model = model_config(
        **params,
    )
    preds = None

    if model_name == "proposed_MTGB":
        model(np.column_stack((x, task)), y, task_info=-1)
    elif model_name == "MTGB":
        model(np.column_stack((x, task)), y)
    elif model_name == "GB":
        if data_pooling:
            if all_features is True:
                x = np.column_stack((x, task))
                model(x, y)
            else:
                model(x, y)
        else:
            x = np.column_stack((x, task))
            x, t = split_task(x, -1)
            unique = np.unique(t)
            T = len(unique)
            tasks_dic = dict(zip(unique, range(T)))
            preds = np.zeros_like(y)
            for r_label, _ in tasks_dic.items():
                idx_r = t == r_label
                X_r = x[idx_r]
                y_r = y[idx_r]

                model = model_config(
                    **params,
                )
                model(X_r, y_r)
                preds[idx_r] = model.predict(X_r)

    if preds is None:
        if model_name == "proposed_MTGB":
            preds = model.predict(np.column_stack((x, task)), task_info=-1)
        elif model_name == "MTGB":
            preds = model.predict(np.column_stack((x, task)))
        elif model_name == "GB":
            preds = model.predict(x)
    return x, y, task, preds


model_name = "proposed_MTGB"
x, y, task, preds = model_to_fit(
    model_name=model_name, problem="regression", data_pooling=False, all_features=False
)
scatter(
    x,
    y,
    task,
    preds,
    f"{model_name}, Single task RMSE {mean_squared_error(y, preds, squared=False):.7f}",
)
