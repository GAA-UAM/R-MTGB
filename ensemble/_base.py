import copy
import warnings
import numpy as np
from tqdm import trange
from numbers import Integral
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from abc import abstractmethod
from scipy.special import expit as sigmoid
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import check_random_state
from sklearn.base import _fit_context
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._gradient_boosting import (
    _random_sample_mask,
)
from ._utils import obj as ensemble_pred
from sklearn.tree._tree import DTYPE
from scipy.special import logsumexp
from ._loss_utils import *
from sklearn.utils.validation import (
    check_random_state,
    _check_sample_weight,
)

from scipy.optimize import minimize


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
        n_common_estimators,
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
    ):
        self.n_estimators = n_estimators
        self.n_common_estimators = n_common_estimators
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

        np.random.seed(self.random_state)

    @abstractmethod
    def _encode_y(self, y=None):
        """Called by fit to validate and encode y."""

    @abstractmethod
    def _get_loss(self, sample_weight: np.ndarray):
        """Get loss object from sklearn._loss.loss."""

    def _neg_gradient(
        self, y: np.ndarray, raw_predictions: np.ndarray, task_type: str, i: int, r: int
    ):

        neg_gradient = self._loss_util.negative_gradient(
            y,
            raw_predictions,
        )

        if self.tasks_dic != None:
            # Multi-task learning
            if task_type == "data_pooling":
                # ∂L/∂ch = (∂L/∂obj()).(∂obj()/∂ch)
                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    neg_gradient[idx_r] *= 1 - self.sigmas_[i if i == 0 else i - 1, r]
            else:
                # ∂L/∂r_h= (∂L/∂obj()).(∂obj()/∂rh)
                return neg_gradient
        return neg_gradient

    def _obj_fun(self, theta, ch, rh, y):

        if rh.ndim > 1 and not self.is_classifier:
            rh = rh.squeeze()
            ch = ch.squeeze()

        sigma = sigmoid(theta)
        loss = self._loss_util(y, ensemble_pred(sigma, ch, rh), None)
        grad_theta = self._loss_util.gradient_theta(y, ch, rh, sigma)

        # Finite difference approximation
        epsilon = 1e-3
        theta_plus, theta_minus = theta + epsilon, theta - epsilon
        w_pred_plus = ensemble_pred(sigmoid(theta_plus), ch, rh)
        w_pred_minus = ensemble_pred(sigmoid(theta_minus), ch, rh)
        loss_plus, loss_minus = map(
            lambda w: self._loss_util(y, w, None), [w_pred_plus, w_pred_minus]
        )
        grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
        assert np.allclose(
            np.sum(grad_theta), grad_approx, rtol=1e-2, atol=1e-2
        ), f"Gradient (w.r.t theta) mismatch detected. Analytic: {np.sum(grad_theta)}, Approx: {grad_approx}"

        return loss, np.sum(grad_theta)

    def _opt_theta(self, ch, rh, y, theta):

        theta = np.atleast_1d(theta)
        _, grad = self._obj_fun(theta, ch, rh, y)
        return theta - (self.learning_rate * grad)

    def _fit_stage(
        self,
        i,
        r,
        X,
        y,
        raw_predictions,
        sample_mask,
        sample_weight,
        task_type=None,
    ):
        for k in range(self._loss_util.K):

            neg_gradient = self._neg_gradient(y, raw_predictions, task_type, i, r)

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                min_impurity_decrease=0.0,
                max_features=self.max_features,
                max_leaf_nodes=2,
                random_state=self._rng,
                ccp_alpha=0.0,
            )

            if self.subsample < 1.0:
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            tree.fit(X, neg_gradient, sample_weight=sample_weight, check_input=False)

            raw_prediction = self._loss_util.update_terminal_regions(
                tree.tree_,
                X,
                y,
                neg_gradient,
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            # Shift the index for specific tasks, leaving the common task at index 0
            r = r + 1 if task_type == "specific_task" else r
            self.estimators_[i, r] = tree
            if self.tasks_dic is None:
                self.residual_[i, :] = np.abs(neg_gradient).mean(axis=0)
            else:
                self.residual_[i, r, :] = np.abs(neg_gradient).mean(axis=0)
        return raw_prediction

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

    def _fit_initial_model(self, X, y, r):
        if not hasattr(self, "inits_"):
            self.inits_ = np.empty((self.T + 1,), dtype=object)
        if not hasattr(self, "init_"):
            self.init_ = self.init
        else:
            del self.init_
            self.init_ = self.init
        if self.init_ is None:
            if self.is_classifier:
                self.init_ = DummyClassifier(strategy="prior")
            else:
                # self.init_ = DummyRegressor(strategy="constant", constant=0)
                self.init_ = DummyRegressor(strategy="mean")
        self.init_.fit(X, y)
        self.inits_[r,] = copy.deepcopy(self.init_)
        return _init_raw_predictions(X, self.init_, self._loss, self.is_classifier)

    def _init_state(self, y):
        self.estimators_ = np.empty((self.n_estimators, self.T + 1), dtype=object)
        if not self.is_classifier:
            if y.ndim < 2:
                y = y[:, np.newaxis]
            num_cols = y.shape[1]
        elif self.is_classifier:
            num_cols = len(np.unique(y))

        self.theta_ = np.zeros((self.n_estimators, self.T), dtype=np.float64)
        self.sigmas_ = np.zeros_like(self.theta_, dtype=np.float64)

        shape = (
            (self.n_estimators, self.T + 1, num_cols)
            if self.tasks_dic is not None
            else (self.n_estimators, num_cols)
        )
        self.residual_ = np.zeros(shape, dtype=np.float32)

        self.train_score_ = np.zeros((self.n_estimators, self.T), dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, "train_score_"):
            del self.train_score_
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

        self._loss = self._get_loss(sample_weight=sample_weight)

        if self.is_classifier:
            self._loss_util = CE(len(set(y_train)))
        else:
            self._loss_util = MSE()

        if task_info is not None:
            # Multi-task learning
            X_train, self.t = self._split_task(X_train, task_info)
            unique = np.unique(self.t)
            self.T = len(unique)
            self.tasks_dic = dict(zip(unique, range(self.T)))
            ch = self._fit_initial_model(X_train, y_train, 0)
            rh = np.zeros_like(ch)

            for r_label, r in self.tasks_dic.items():
                idx_r = self.t == r_label
                X_r = X_train[idx_r]
                y_r = y_train[idx_r]
                rh[idx_r] = self._fit_initial_model(X_r, y_r, r + 1)

        else:
            self.T = 0
            self.tasks_dic = None
            rh = None
            ch = self._fit_initial_model(X_train, y_train, 0)

        self._set_max_features()
        self._init_state(y_train)

        self._rng = check_random_state(self.random_state)

        n_stages = self._fit_stages(
            X_train,
            y_train,
            ch,
            rh,
            sample_weight_train,
            sample_weight_val,
            X_val,
            y_val,
            task_info,
        )

        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            self.sigmas_ = self.sigmas_[:n_stages]
            self.theta_ = self.theta_[:n_stages]
            self.residual_ = self.residual_[:n_stages]
            self.n_estimators = n_stages

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

    def _subsampling(self, X):
        n_samples = X.shape[0]
        sample_mask = np.ones((n_samples,), dtype=bool)
        if self.subsample < 1.0:
            n_inbag = max(1, int(self.subsample * n_samples))
            sample_mask = _random_sample_mask(n_samples, n_inbag, self._rng)

        return sample_mask

    def _track_loss(
        self,
        i,
        X,
        y,
        sample_weight,
        raw_predictions,
    ):

        if self.subsample < 1.0:
            if self.tasks_dic is None:
                sample_mask = self._subsampling(X)
                self.train_score_[i,] = self._loss_util(
                    y=y[sample_mask],
                    raw_predictions=raw_predictions[sample_mask],
                    sample_weight=sample_weight[sample_mask],
                )

            elif self.tasks_dic is not None:

                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    sample_mask = self._subsampling(X[idx_r])
                    self.train_score_[i, r] = self._loss_util(
                        y=y[idx_r][sample_mask],
                        raw_predictions=raw_predictions[idx_r][sample_mask],
                        sample_weight=sample_weight[idx_r][sample_mask],
                    )

        else:
            if self.tasks_dic is None:
                self.train_score_[i,] = self._loss_util(
                    y=y,
                    raw_predictions=raw_predictions,
                    sample_weight=sample_weight,
                )
            elif self.tasks_dic != None:
                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    self.train_score_[i, r] = self._loss_util(
                        y=y[idx_r],
                        raw_predictions=raw_predictions[idx_r],
                        sample_weight=sample_weight[idx_r],
                    )

    def _update_prediction(self, ch, rh, sigma):

        raw_predictions = np.zeros_like(ch)
        for r_label, r in self.tasks_dic.items():
            idx_r = self.t == r_label
            raw_predictions[idx_r] = ensemble_pred(sigma[r], ch[idx_r], rh[idx_r])
        return raw_predictions

    def _task_theta_opt(self, y, ch, rh, i):

        for r_label, r in self.tasks_dic.items():
            idx_r = self.t == r_label
            y_r = y[idx_r]

            theta = self._opt_theta(
                ch[idx_r],
                rh[idx_r],
                y_r,
                self.theta_[i, r],
            )

            self.theta_[i, r] = theta[0]
            self.sigmas_[i, r] = sigmoid(theta[0])

    def _fit_stages(
        self,
        X,
        y,
        ch,
        rh,
        sample_weight,
        sample_weight_val,
        X_val,
        y_val,
        task_info,
    ) -> np.int32:

        y = self._label_y(y)
        self.sigmas_[0, :] = sigmoid(self.theta_[0, :])

        if self.tasks_dic != None:
            raw_predictions = (
                self._update_prediction(ch, rh, sigmoid(self.theta_[0, :]))
                if task_info
                else ch
            )
        else:
            raw_predictions = ch

        if self.early_stopping is not None:
            loss_history = np.full(self.early_stopping, np.inf)
            # creating a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val, task_info)
            y_val = self._label_y(y_val)

        boosting_bar = trange(
            0,
            self.n_estimators,
            leave=True,
            desc="Boosting epochs",
            dynamic_ncols=True,
        )

        for i in boosting_bar:

            # Multi-task learning
            if self.tasks_dic != None:

                if i < self.n_common_estimators:
                    # Common tasks

                    ch = self._fit_stage(
                        i,
                        0,
                        X,
                        y,
                        raw_predictions,
                        self._subsampling(X),
                        sample_weight,
                        "data_pooling",
                    )

                    self._task_theta_opt(y, ch, rh, i)

                    raw_predictions = self._update_prediction(
                        ch,
                        rh,
                        self.sigmas_[i, :],
                    )

                    self._track_loss(
                        i,
                        X,
                        y,
                        sample_weight,
                        raw_predictions,
                    )

                else:

                    for r_label, r in self.tasks_dic.items():
                        idx_r = self.t == r_label
                        X_r = X[idx_r]
                        y_r = y[idx_r]
                        sample_mask_r = self._subsampling(X_r)

                        rh[idx_r] = self._fit_stage(
                            i,
                            r,
                            X_r,
                            y_r,
                            raw_predictions[idx_r],
                            sample_mask_r,
                            sample_weight[idx_r],
                            "specific_task",
                        )

                    self._task_theta_opt(y, ch, rh, i)

                    raw_predictions = self._update_prediction(
                        ch,
                        rh,
                        self.sigmas_[i, :],
                    )

                    self._track_loss(
                        i,
                        X,
                        y,
                        sample_weight,
                        raw_predictions,
                    )

            else:
                raw_predictions = self._fit_stage(
                    i,
                    0,
                    X,
                    y,
                    raw_predictions,
                    self._subsampling(X),
                    sample_weight,
                )

                self._track_loss(
                    i,
                    X,
                    y,
                    sample_weight,
                    raw_predictions,
                )

            if self.verbose > 1:
                boosting_bar.set_description(
                    f"Loss: {self.train_score_[i]:.4f}", refresh=True
                )

            if self.early_stopping is not None and i > 0:
                validation_loss = self._loss_util(
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

            self.X_test, self.t_test = self._split_task(X, task_info)
            t = self.t_test
            unique = np.unique(t)
            T_test = len(unique)
            self.tasks_dic_test = dict(zip(unique, range(T_test)))

            # 0_th index (common task estimator)
            self.X_test = self.estimators_[0, 0]._validate_X_predict(
                self.X_test, check_input=True
            )

            ch = _init_raw_predictions(
                self.X_test, self.inits_[0], self._loss, self.is_classifier
            )

            rh = np.zeros_like(ch)

            for r_label, r in self.tasks_dic_test.items():
                idx_r = t == r_label
                X_r = self.X_test[idx_r]
                X_r = self.estimators_[
                    self.n_common_estimators, int(self.tasks_dic[r_label]) + 1
                ]._validate_X_predict(X_r, check_input=True)

                rh[idx_r] = _init_raw_predictions(
                    X_r, self.inits_[r + 1], self._loss, self.is_classifier
                )
            
        elif task_info is None:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
            X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

            ch = _init_raw_predictions(X, self.init_, self._loss, self.is_classifier)

            rh = None

        return ch, rh

    def _raw_predict(self, X: np.ndarray, task_info=None) -> np.ndarray:
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        ch, rh = self._raw_predict_init(X, task_info)
        raw_predictions = self._predict_stages(self.estimators_, X, ch, rh, task_info)
        return raw_predictions

    def _predict_stages(
        self,
        estimators_: np.ndarray,
        X: np.ndarray,
        ch: np.ndarray,
        rh: np.ndarray,
        task_info=None,
    ) -> np.ndarray:

        if not self.is_classifier:
            ch = ch.squeeze()
            if task_info:
                rh = rh.squeeze()

        # Multi task learning
        if self.tasks_dic != None and task_info != None:

            t = self.t_test

            for i in range(len(estimators_)):
                if i < self.n_common_estimators:
                    # Update ommon task prediction
                    tree = estimators_[i, 0]
                    ch += self.learning_rate * tree.predict(self.X_test)

                    del tree

                    for r_label, r in self.tasks_dic_test.items():
                        if r_label not in self.tasks_dic:
                            raise ValueError(
                                "The task {} was not present in the training set".format(
                                    r_label
                                )
                            )
                        idx_r = t == r_label

                        ch[idx_r] = ensemble_pred(
                            (self.sigmas_[i, r]),
                            ch[idx_r],
                            rh[idx_r],
                        )

                else:
                    # Update task-specific predictions
                    for r_label, r in self.tasks_dic_test.items():
                        if r_label not in self.tasks_dic:
                            raise ValueError(
                                "The task {} was not present in the training set".format(
                                    r_label
                                )
                            )

                        tree = estimators_[i, r + 1]
                        idx_r = t == r_label

                        X_r = self.X_test[idx_r]
                        rh[idx_r] += self.learning_rate * tree.predict(X_r)

                        ch[idx_r] = ensemble_pred(
                            (self.sigmas_[i, r]),
                            ch[idx_r],
                            rh[idx_r],
                        )

                        del tree

        else:
            # Single task learning
            for i in range(len(estimators_)):
                tree = estimators_[i][0]
                ch += self.learning_rate * tree.predict(X)

        return ch

    def _staged_raw_predict(self, X: np.ndarray, task_info=-1):

        ch, rh = self._raw_predict_init(X, task_info)

        if not self.is_classifier:
            ch = ch.squeeze()
            rh = rh.squeeze()

        # Multi task learning.
        if self.tasks_dic != None and task_info != None:

            X, t = self._split_task(X, task_info)
            unique = np.unique(t)
            tasks_dic = dict(zip(unique, range(len(unique))))

            for i in range(len(self.estimators_)):
                if i < self.n_common_estimators:
                    # Common task prediction.
                    tree = self.estimators_[i, 0]
                    ch += tree.predict(X) * self.learning_rate

                    del tree

                    for r_label, r in self.tasks_dic_test.items():
                        if r_label not in self.tasks_dic:
                            raise ValueError(
                                "The task {} was not present in the training set".format(
                                    r_label
                                )
                            )
                        idx_r = t == r_label

                        ch[idx_r] = ensemble_pred(
                            (self.sigmas_[i, r]),
                            ch[idx_r],
                            rh[idx_r],
                        )

                else:
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
                        rh[idx_r] += self.learning_rate * tree.predict(X_r)

                        ch[idx_r] = ensemble_pred(
                            (self.sigmas_[i, r]),
                            ch[idx_r],
                            rh[idx_r],
                        )

                        del tree

        else:
            # Single task learning.
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
            for i in range(len(self.estimators_)):
                tree = self.estimators_[i][0]
                ch += self.learning_rate * tree.predict(X)

        yield ch.copy()

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
