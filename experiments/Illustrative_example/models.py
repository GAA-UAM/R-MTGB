import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import warnings


class model_config:
    def __init__(
        self,
        problem,
        model,
        max_depth,
        n_estimators,
        n_common_estimators,
        learning_rate,
        random_state,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_common_estimators = n_common_estimators
        self.random_state = random_state

        warnings.simplefilter("ignore")
        np.random.seed(self.random_state)

        model_params = {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "max_features": None,
            "subsample": 1.0,
        }

        additional_params = {
            "n_common_estimators": self.n_common_estimators,
        }

        if model == "GB":
            if problem == "regression":
                self.model = GradientBoostingRegressor(**model_params)
            else:
                self.model = GradientBoostingClassifier(**model_params)
        elif model == "MTGB":
            sys.path.append(r"D:\Ph.D\Programming\Py\MT-GB\MT_GB")
            from model.mtgb import MTGBRegressor, MTGBClassifier

            if problem == "regression":
                self.model = MTGBRegressor(**model_params, **additional_params)
            else:
                self.model = MTGBClassifier(**model_params, **additional_params)
        elif model == "proposed_MTGB":
            sys.path.append(r"D:\Ph.D\Programming\Py\NoiseAwareBoost")
            from mtgb_models.mt_gb import MTGBRegressor, MTGBClassifier

            if problem == "regression":
                self.model = MTGBRegressor(**model_params, **additional_params)
            else:
                self.model = MTGBClassifier(**model_params, **additional_params)

    def __call__(self, X, y, task_info=None):
        if task_info is not None:
            self.model.fit(X, y, task_info)
        else:
            self.model.fit(X, y)

    def predict(self, X, task_info=None):
        if task_info is not None:
            return self.model.predict(X, task_info)
        else:
            return self.model.predict(X)
