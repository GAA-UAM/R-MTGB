import numpy as np
from Ensemble._base import BaseMTGB
from sklearn.tree._tree import DOUBLE
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets


class RMTGBClassifier(BaseMTGB):
    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_iter_1st=100,
        n_iter_2nd=1,
        n_iter_3rd=1,
        subsample=1.0,
        criterion="squared_error",
        max_depth=3,
        init=None,
        random_state=None,
        max_features=None,
        validation_fraction=0.1,
        early_stopping=None,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_iter_1st=n_iter_1st,
            n_iter_2nd=n_iter_2nd,
            n_iter_3rd=n_iter_3rd,
            criterion=criterion,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            validation_fraction=validation_fraction,
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

    def decision_function(self, X):

        raw_predictions = self._predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        if raw_predictions.ndim == 1:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
        return self.classes_[encoded_classes]

    def predict_proba(self, X):
        raw_predictions = self.decision_function(X)
        return self._loss_util.predict_proba(raw_predictions)

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(proba)


class RMTGBRegressor(BaseMTGB):
    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_iter_1st=100,
        n_iter_2nd=1,
        n_iter_3rd=1,
        subsample=1.0,
        criterion="squared_error",
        max_depth=3,
        init=None,
        random_state=None,
        max_features=None,
        validation_fraction=0.1,
        early_stopping=None,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_iter_1st=n_iter_1st,
            n_iter_2nd=n_iter_2nd,
            n_iter_3rd=n_iter_3rd,
            criterion=criterion,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            validation_fraction=validation_fraction,
            early_stopping=early_stopping,
        )

    def _encode_y(self, y=None):
        self.is_classifier = False
        self.n_trees_per_iteration_ = 1
        y = y.astype(DOUBLE, copy=False)
        return y

    def _get_loss(self, sample_weight):
        pass

    def predict(self, X):
        return self._predict(X)
