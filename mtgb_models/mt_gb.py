import numpy as np
from ensemble._base import BaseMTGB
from sklearn.tree._tree import DOUBLE
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets


class MTGBClassifier(BaseMTGB):
    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        n_common_estimators=1,
        subsample=1.0,
        criterion="squared_error",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        ccp_alpha=0.0,
        early_stopping=None,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_common_estimators=n_common_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            ccp_alpha=ccp_alpha,
            early_stopping=early_stopping,
        )

    def _encode_y(self, y):

        check_classification_targets(y)

        label_encoder = LabelEncoder()

        encoded_y_int = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]

        self.n_trees_per_iteration_ = 1
        encoded_y = encoded_y_int.astype(np.int64, copy=False)

        self.n_classes_ = n_classes
        self.is_classifier = True

        n_trim_classes = n_classes

        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        return encoded_y

    def _get_loss(self, sample_weight):
        pass

    def decision_function(self, X, task_info=None):

        raw_predictions = self._raw_predict(
            X,
            task_info,
        )
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        yield from self._staged_raw_predict(X)

    def predict(self, X, task_info=None):
        raw_predictions = self.decision_function(
            X,
            task_info,
        )
        if raw_predictions.ndim == 1:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
        return self.classes_[encoded_classes]

    def staged_predict(self, X, task_info=None):
        for raw_predictions in self._staged_raw_predict(X, task_info):
            encoded_classes = np.argmax(raw_predictions, axis=1)
            yield self.classes_.take(encoded_classes, axis=0)

    def predict_proba(self, X, task_info=None):
        raw_predictions = self.decision_function(X, task_info=None)
        return self._loss_util.predict_proba(raw_predictions)

    def predict_log_proba(self, X, task_info=None):
        proba = self.predict_proba(X, task_info=None)
        return np.log(proba)

    def staged_predict_proba(self, X):
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self._loss_util.predict_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e


class MTGBRegressor(BaseMTGB):
    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        n_common_estimators=1,
        subsample=1.0,
        criterion="squared_error",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        ccp_alpha=0.0,
        early_stopping=None,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_common_estimators=n_common_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            ccp_alpha=ccp_alpha,
            early_stopping=early_stopping,
        )

    def _encode_y(self, y=None):
        self.is_classifier = False
        self.n_trees_per_iteration_ = 1
        y = y.astype(DOUBLE, copy=False)
        return y

    def _get_loss(self, sample_weight):
        pass

    def predict(self, X, task_info=None):
        return self._raw_predict(X, task_info)

    def staged_predict(self, X, task_info=None):

        for raw_predictions in self._staged_raw_predict(X, task_info):
            yield raw_predictions.ravel()
