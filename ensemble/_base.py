import warnings
import numpy as np
from numbers import Integral
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn._loss.loss import (
    HalfBinomialLoss,
    HalfSquaredError,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from abc import abstractmethod
from scipy.special import expit as sigmoid
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import check_random_state
from sklearn.base import _fit_context
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csc_matrix, csr_matrix, issparse
from sklearn.ensemble._gradient_boosting import (
    _random_sample_mask,
)

from sklearn.tree._tree import DTYPE
from scipy.special import logsumexp
from ._loss_utils import *
from sklearn.utils.validation import (
    check_random_state,
    _check_sample_weight,
)


def _init_raw_predictions(X, estimator, loss, is_classifier):
    if is_classifier:
        predictions = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)


class BaseMTGB(BaseGradientBoosting):
    @abstractmethod
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_estimators,
        criterion,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_depth,
        min_impurity_decrease,
        init,
        subsample,
        max_features,
        ccp_alpha,
        random_state,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        early_stopping=None,
        step_size=1
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.is_classifier = False
        self.early_stopping = early_stopping
        self.step_size = step_size

    @abstractmethod
    def _encode_y(self, y=None):
        """Called by fit to validate and encode y."""

    @abstractmethod
    def _get_loss(self, sample_weight: np.ndarray):
        """Get loss object from sklearn._loss.loss."""

    def _neg_gradient(
        self,
        y: np.ndarray,
        raw_predictions: np.ndarray,
        task_type: str,
        theta: np.ndarray,
    ):

        neg_gradient = self._aux_loss.negative_gradient(
            y,
            raw_predictions,
        )

        sigma = sigmoid(theta)
        target_type = type_of_target(y)
        if (
            sigma.ndim > 1
            and not self.is_classifier
            and target_type not in {"continuous-multioutput", "multiclass-multioutput"}
        ):
            sigma = sigma.squeeze()

        if task_type == "common_task":
            neg_gradient = sigma * neg_gradient
        else:
            neg_gradient = (1 - sigma) * neg_gradient

        return neg_gradient

    def _task_obj_fun(self, theta, c_h, r_h, y):
        if r_h.ndim > 1 and not self.is_classifier:
            r_h = r_h.squeeze()
            c_h = c_h.squeeze()
        # As fmin_l_bfgs_b flattens the input
        # parameters into a 1D array.
        if self.is_classifier:
            theta = theta.reshape(c_h.shape)
        w_pred = (sigmoid(theta) * c_h) + ((1 - sigmoid(theta)) * r_h)
        loss = self._aux_loss(y, w_pred, None)

        # L2 regularization
        # penalizing rapid changes between consecutive theta values.
        # regularization_term = self.step_size * np.sum(
        #     np.diff(theta) ** 2
        # )  # Smoothness penalty (second-order difference)
        # loss += regularization_term

        # Gradient of the loss w.r.t theta (using chain rule)
        # Apply the chain rule: ∂theta = (∂L/∂w_pred) * (∂w_pred/∂theta)
        # (∂w_pred/∂theta) = sigmoid(theta) * (1 - sigmoid(theta)) * (c_h - r_h)
        grad_theta = self._aux_loss.negative_gradient_theta(y, c_h, r_h, sigmoid(theta))

        # L2 regularization
        # grad_reg = (
        #     -2
        #     * self.step_size
        #     * (np.diff(np.hstack(([0], theta))) - np.diff(np.hstack((theta, [0]))))
        # )
        # grad_theta += grad_reg

        if self.is_classifier:
            grad_theta = grad_theta.flatten()

        return loss, grad_theta

    def _opt_theta(self, c_h, r_h, y):

        args = (c_h, r_h, y)
        result = fmin_l_bfgs_b(
            self._task_obj_fun,
            # np.random.uniform(-1.0, 1.0, c_h.shape),
            np.random.normal(0, 0.1, c_h.shape),
            args=args,
            approx_grad=False,
            maxls=1,
            factr=1e7,  # Convergence criterion
            pgtol=1e-5,
        )
        optimized_theta = result[0][0]
        return optimized_theta

    def _fit_stage(
        self,
        i,
        r,
        X,
        y,
        raw_predictions,
        sample_mask,
        random_state,
        sample_weight,
        learning_rate,
        theta,
        task_type,
        X_csc=None,
        X_csr=None,
    ):
        for k in range(self._aux_loss.K):

            neg_gradient = self._neg_gradient(y, raw_predictions, task_type, theta)

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha,
            )

            if self.subsample < 1.0:
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csc if X_csc is not None else X
            tree.fit(X, neg_gradient, sample_weight=sample_weight, check_input=False)

            X_for_tree_update = X_csr if X_csr is not None else X
            rawpredictions = self._aux_loss.update_terminal_regions(
                tree.tree_,
                X_for_tree_update,
                y,
                neg_gradient,
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=learning_rate,
                k=k,
            )

            self.estimators_[i, r] = tree
            self.residual_[i, r] = np.abs(neg_gradient).mean(axis=0)

        return rawpredictions

    def _set_max_features(self):
        """Set self.max_features_."""
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if self.is_classifier:
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            else:  # self.max_features == "log2"
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:  # float
            max_features = max(1, int(self.max_features * self.n_features_in_))

        self.max_features_ = max_features

    def _init_theta(self, num_rows, num_cols):
        return np.zeros((self.n_estimators + 1, num_rows, num_cols), dtype=np.float64)

    def _init_state(self, y):
        self.init_ = self.init
        if self.init_ is None:
            if self.is_classifier:
                self.init_ = DummyClassifier(strategy="prior")

            else:
                self.init_ = DummyRegressor(strategy="mean")

        self.estimators_ = np.empty((self.n_estimators, self.T + 1), dtype=object)
        self.residual_ = np.zeros((self.n_estimators, self.T + 1), dtype=np.float32)
        self.sigmas_ = np.zeros((self.n_estimators, self.T), dtype=np.float64)
        if not self.is_classifier:
            if y.ndim < 2:
                y = y[:, np.newaxis]
            num_rows, num_cols = y.shape
        elif self.is_classifier:
            num_cols = len(np.unique(y))
            num_rows = y.shape[0]
        self.theta_ = self._init_theta(num_rows, num_cols)

        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        self.train_score_r = np.zeros((self.n_estimators, self.T), dtype=np.float64)
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_scores_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_score_ = np.nan

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, "train_score_"):
            del self.train_score_
        if hasattr(self, "oob_improvement_"):
            del self.oob_improvement_
        if hasattr(self, "oob_scores_"):
            del self.oob_scores_
        if hasattr(self, "oob_score_"):
            del self.oob_score_
        if hasattr(self, "init_"):
            del self.init_
        if hasattr(self, "_rng"):
            del self._rng

    def _is_fitted(self):
        return len(getattr(self, "estimators_", [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    @_fit_context(
        # GradientBoosting*.init is not validated yet
        prefer_skip_nested_validation=False
    )
    def _split_task(self, X: np.ndarray, task_info: int) -> np.ndarray:

        # Ensuring that every distinct value is replaced with an index, beginning from zero.
        unique_values = np.unique(X[:, task_info])
        mapping = {value: index for index, value in enumerate(unique_values)}
        X[:, task_info] = [mapping[value] for value in X[:, task_info]]

        # Separating the input array and task indices.
        X_task = X[:, task_info]
        X_data = np.delete(X, task_info, axis=1).astype(float)
        return X_data, X_task

    def _stratified_train_test_split(
        self, X, y, sample_weight, test_size, random_state
    ):

        stratify = y if self.is_classifier else None
        X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val = (
            train_test_split(
                X,
                y,
                sample_weight,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        )

        if self.is_classifier:
            # Check for missing classes in y_val
            unique_classes = np.unique(y)
            missing_classes = np.setdiff1d(unique_classes, np.unique(y_val))

            if len(missing_classes) > 0:
                # Find indices of the missing classes in the original dataset
                missing_indices = [np.where(y == cls)[0][0] for cls in missing_classes]

                # Move these samples from train to val
                for idx in missing_indices:
                    X_val = np.vstack([X_val, X[idx].reshape(1, -1)])
                    y_val = np.append(y_val, y[idx])
                    sample_weight_val = np.append(sample_weight_val, sample_weight[idx])

                    train_idx = np.where((X_train == X[idx]).all(axis=1))[0][0]
                    X_train = np.delete(X_train, train_idx, axis=0)
                    y_train = np.delete(y_train, train_idx)
                    sample_weight_train = np.delete(sample_weight_train, train_idx)

        return (
            X_train,
            X_val,
            y_train,
            y_val,
            sample_weight_train,
            sample_weight_val,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, task_info=None):
        if not self.warm_start:
            self._clear_state()

        sample_weight = _check_sample_weight(None, X)

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=DTYPE,
            multi_output=True,
        )

        y = self._encode_y(y=y)

        if self.early_stopping is not None:
            (
                X_train,
                X_val,
                y_train,
                y_val,
                sample_weight_train,
                sample_weight_val,
            ) = self._stratified_train_test_split(
                X, y, sample_weight, self.validation_fraction, self.random_state
            )
        else:
            X_train, y_train, sample_weight_train = X, y, sample_weight
            X_val = y_val = sample_weight_val = None

        if task_info is not None:
            X_train, self.t = self._split_task(X_train, task_info)
            unique = np.unique(self.t)
            self.T = len(unique)
            self.tasks_dic = dict(zip(unique, range(self.T)))
        else:
            self.T = 0
            self.tasks_dic = None

        self._loss = self._get_loss(sample_weight=sample_weight)

        if self.is_classifier:
            self._aux_loss = CE(len(set(y_train)))
        else:
            self._aux_loss = MSE()

        self._init_state(y)
        self._set_max_features()

        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X_train.shape[0], 1),
                dtype=np.float64,
            )
        else:
            self.init_.fit(X_train, y_train)

            raw_predictions = _init_raw_predictions(
                X_train, self.init_, self._loss, self.is_classifier
            )

        self._rng = check_random_state(self.random_state)

        n_stages = self._fit_stages(
            X_train,
            y_train,
            raw_predictions,
            sample_weight_train,
            sample_weight_val,
            self._rng,
            X_val,
            y_val,
            task_info,
        )

        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            self.train_score_r = self.train_score_r[:n_stages]
            self.sigmas_ = self.sigmas_[:n_stages]
            self.theta_ = self.theta_[:n_stages]
            self.residual_ = self.residual_[:n_stages]
            self.n_estimators = n_stages
            if hasattr(self, "oob_improvement_"):
                # OOB scores were computed
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
                self.oob_scores_ = self.oob_scores_[:n_stages]
                self.oob_score_ = self.oob_scores_[-1]

        return self

    def _label_y(self, y: np.ndarray):
        if self._loss.is_multiclass:
            y = LabelBinarizer().fit_transform(y)
        elif type_of_target(y) == "binary":
            Y = np.zeros((y.shape[0], 2), dtype=np.float64)
            if y.ndim == 2:
                y = y.squeeze()
            for k in range(2):
                Y[:, k] = y == k
            y = Y
        elif not self.is_classifier:
            return y
        return y

    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_weight_val,
        random_state,
        X_val,
        y_val,
        task_info,
    ) -> np.int32:
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if isinstance(
            self._loss,
            (
                HalfSquaredError,
                HalfBinomialLoss,
            ),
        ):
            factor = 2
        else:
            factor = 1

        y = self._label_y(y)

        if self.early_stopping is not None:
            loss_history = np.full(self.early_stopping, np.inf)
            # creating a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val, task_info)
            y_val = self._label_y(y_val)

        sample_weight_c = _check_sample_weight(sample_weight, X)

        # Initializing theta.
        self.theta_[0, :, :] = np.random.uniform(-1.0, 1.0, raw_predictions.shape)

        for i in range(0, self.n_estimators):
            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                y_oob_masked = y[~sample_mask]
                sample_weight_oob_masked = sample_weight_c[~sample_mask]
                if i == 0:  # store the initial loss to compute the OOB score
                    initial_loss = factor * self._aux_loss(
                        y=y_oob_masked,
                        raw_predictions=raw_predictions[~sample_mask],
                        sample_weight=sample_weight_oob_masked,
                    )

            # Common task.
            raw_predictions_c = self._fit_stage(
                i,
                0,
                X,
                y,
                raw_predictions.copy(),
                sample_mask,
                random_state,
                sample_weight_c,
                self.learning_rate,
                self.theta_[i, :, :],
                "common_task",
                X_csc=X_csc,
                X_csr=X_csr,
            )

            # Task specific.
            indices = []
            if self.tasks_dic != None:
                predictions = np.zeros_like(raw_predictions.copy())
                raw_predictions_r = np.zeros_like(raw_predictions.copy())
                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    X_r = X[idx_r]
                    y_r = y[idx_r]
                    sample_weight_r = _check_sample_weight(sample_weight[idx_r], X_r)

                    raw_predictions_r[idx_r] = self._fit_stage(
                        i,
                        r + 1,
                        X_r,
                        y_r,
                        raw_predictions.copy()[idx_r],
                        sample_mask[idx_r],
                        random_state,
                        sample_weight_r,
                        self.learning_rate,
                        self.theta_[i, idx_r, :],
                        "specific_task",
                        X_csc=X_csc,
                        X_csr=X_csr,
                    )

                    theta = self._opt_theta(
                        raw_predictions_c[idx_r],
                        raw_predictions_r[idx_r],
                        y_r,
                    )

                    self.theta_[i + 1, idx_r, :] = theta
                    sigma = sigmoid(theta) * 1
                    self.sigmas_[i, r] = sigma
                    predictions[idx_r] = (sigma * raw_predictions_c[idx_r]) + (
                        (1 - sigma) * raw_predictions_r[idx_r]
                    )

                    indices.append(idx_r)

            else:
                predictions = raw_predictions_c
            raw_predictions = predictions

            # track loss
            if do_oob:
                self.train_score_[i] = factor * self._aux_loss(
                    y=y[sample_mask],
                    raw_predictions=raw_predictions[sample_mask],
                    sample_weight=sample_weight_c[sample_mask],
                )

                for task in range(self.T):
                    self.train_score_r[i, task] = factor * self._aux_loss(
                        y=y[indices[task]][sample_mask[indices[task]]],
                        raw_predictions=raw_predictions_r[indices[task]][
                            sample_mask[indices[task]]
                        ],
                        sample_weight=sample_weight_c[indices[task]][
                            sample_mask[indices[task]]
                        ],
                    )
                self.oob_scores_[i] = factor * self._aux_loss(
                    y=y_oob_masked,
                    raw_predictions=raw_predictions[~sample_mask],
                    sample_weight=sample_weight_c[~sample_mask],
                )
                previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                self.train_score_[i] = factor * self._aux_loss(
                    y=y,
                    raw_predictions=raw_predictions,
                    sample_weight=sample_weight_c,
                )
                for task in range(self.T):
                    self.train_score_r[i, task] = factor * self._aux_loss(
                        y=y[indices[task]],
                        raw_predictions=raw_predictions_r[indices[task]],
                        sample_weight=sample_weight_c[indices[task]],
                    )

            if self.early_stopping is not None and i > 0:
                validation_loss = factor * self._aux_loss(
                    y_val, next(y_val_pred_iter), sample_weight_val
                )

                if np.any(validation_loss + 1e-4 < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break
        return i + 1

    def _raw_predict_init(self, X: np.ndarray, task_info=None) -> np.ndarray:
        """Check input and compute raw predictions of the init estimator."""

        self._check_initialized()
        if task_info is not None:

            X, t = self._split_task(X, task_info)
            unique = np.unique(t)

            # 0_th index (common task estimator)
            X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

            for r_label in unique:
                idx_r = t == r_label
                X[idx_r] = self.estimators_[
                    0, int(self.tasks_dic[r_label]) + 1
                ]._validate_X_predict(X[idx_r], check_input=True)

        elif task_info is None:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
            X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == "zero":

            raw_predictions = np.zeros(
                shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64
            )
        else:
            raw_predictions = _init_raw_predictions(
                X, self.init_, self._loss, self.is_classifier
            )

        return raw_predictions

    def _raw_predict(self, X: np.ndarray, task_info=None) -> np.ndarray:
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X, task_info)
        raw_predictions = self._predict_stages(
            self.estimators_, X, raw_predictions, task_info
        )
        return raw_predictions

    def _predict_stages(
        self,
        estimators_: np.ndarray,
        X: np.ndarray,
        raw_predictions: np.ndarray,
        task_info=None,
    ) -> np.ndarray:

        if not self.is_classifier:
            raw_predictions = raw_predictions.squeeze()

        # Multi task learning.
        if self.tasks_dic != None and task_info != None:
            raw_predictions_r = np.zeros_like(raw_predictions)
            predictions = np.zeros_like(raw_predictions)

            X, t = self._split_task(X, task_info)
            unique = np.unique(t)
            tasks_dic = dict(zip(unique, range(len(unique))))

            for i in range(len(estimators_)):

                # Common task prediction.
                tree = estimators_[i, 0]
                raw_predictions_c = tree.predict(X)

                # Task specific prediction.
                for r_label, r in tasks_dic.items():
                    if r_label not in self.tasks_dic:
                        raise ValueError(
                            "The task {} was not present in the training set".format(
                                r_label
                            )
                        )
                    tree = estimators_[i, r + 1]
                    idx_r = t == r_label
                    X_r = X[idx_r]
                    raw_predictions_r[idx_r] = tree.predict(X_r)
                    predictions[idx_r] = (
                        self.sigmas_[i, r] * raw_predictions_c[idx_r]
                    ) + ((1 - self.sigmas_[i, r]) * raw_predictions_r[idx_r])

                    del tree

                raw_predictions += self.learning_rate * predictions

        # Single task learning.
        else:
            for i in range(len(estimators_)):
                tree = estimators_[i][0]
                raw_predictions += self.learning_rate * tree.predict(X)

        return raw_predictions

    def _staged_raw_predict(self, X: np.ndarray, task_info=None):

        raw_predictions = self._raw_predict_init(X, task_info)
        if raw_predictions.shape[1] == 1:
            raw_predictions = np.squeeze(raw_predictions)

        # Multi task learning.
        if self.tasks_dic != None and task_info != None:
            raw_predictions_r = np.zeros_like(raw_predictions)
            predictions = np.zeros_like(raw_predictions)

            X, t = self._split_task(X, task_info)
            unique = np.unique(t)
            tasks_dic = dict(zip(unique, range(len(unique))))

            for i in range(len(self.estimators_)):

                # Common task prediction.
                tree = self.estimators_[i, 0]
                raw_predictions_c = tree.predict(X)

                # Task specific prediction.
                for r_label, r in tasks_dic.items():
                    if r_label not in self.tasks_dic:
                        raise ValueError(
                            "The task {} was not present in the training set".format(
                                r_label
                            )
                        )
                    tree = self.estimators_[i, r + 1]
                    idx_r = t == r_label
                    X_r = X[idx_r]

                    raw_predictions_r[idx_r] = tree.predict(X_r)
                    predictions[idx_r] = (
                        self.sigmas_[i, r] * raw_predictions_c[idx_r]
                    ) + ((1 - self.sigmas_[i, r]) * raw_predictions_r[idx_r])
                raw_predictions += self.learning_rate * predictions

                yield raw_predictions.copy()

        else:
            # Single task learning.
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
            for i in range(len(self.estimators_)):
                tree = self.estimators_[i][0]
                raw_predictions += self.learning_rate * tree.predict(X)
                yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        self._check_initialized()

        relevant_trees = [
            tree
            for stage in self.estimators_
            for tree in stage
            if tree.tree_.node_count > 1
        ]
        if not relevant_trees:
            # degenerate case where all trees have only one node
            return np.zeros(shape=self.n_features_in_, dtype=np.float64)

        relevant_feature_importances = [
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in relevant_trees
        ]
        avg_feature_importances = np.mean(
            relevant_feature_importances, axis=0, dtype=np.float64
        )
        return avg_feature_importances / np.sum(avg_feature_importances)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        if self.init is not None:
            warnings.warn(
                "Using recursion method with a non-constant init predictor "
                "will lead to incorrect partial dependence values. "
                "Got init=%s." % self.init,
                UserWarning,
            )
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        n_estimators, n_trees_per_stage = self.estimators_.shape
        averaged_predictions = np.zeros(
            (n_trees_per_stage, grid.shape[0]), dtype=np.float64, order="C"
        )
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                tree = self.estimators_[stage, k].tree_
                tree.compute_partial_dependence(
                    grid, target_features, averaged_predictions[k]
                )
        averaged_predictions *= self.learning_rate

        return averaged_predictions

    def apply(self, X):
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves
