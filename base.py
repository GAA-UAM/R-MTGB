from abc import ABCMeta, abstractmethod
import copy
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.stats import _weighted_percentile
import numpy as np
from numbers import Integral, Real
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.ensemble._gradient_boosting import (
    _random_sample_mask,
    predict_stage,
    predict_stages,
)
from sklearn.model_selection import train_test_split
import math
from scipy.sparse import csc_matrix, csr_matrix, issparse
import warnings
from sklearn.utils import check_array, check_random_state, column_or_1d
from time import time
from sklearn.base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from sklearn.tree._tree import DOUBLE, DTYPE, TREE_LEAF
from sklearn._loss.loss import (
    _LOSSES,
    AbsoluteError,
    ExponentialLoss,
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfSquaredError,
    HuberLoss,
    PinballLoss,
)


def _safe_divide(numerator, denominator):
    """Prevents overflow and division by zero."""
    # This is used for classifiers where the denominator might become zero exatly.
    # For instance for log loss, HalfBinomialLoss, if proba=0 or proba=1 exactly, then
    # denominator = hessian = 0, and we should set the node value in the line search to
    # zero as there is no improvement of the loss possible.
    # For numerical safety, we do this already for extremely tiny values.
    if abs(denominator) < 1e-150:
        return 0.0
    else:
        # Cast to Python float to trigger Python errors, e.g. ZeroDivisionError,
        # without relying on `np.errstate` that is not supported by Pyodide.
        result = float(numerator) / float(denominator)
        # Cast to Python float to trigger a ZeroDivisionError without relying
        # on `np.errstate` that is not supported by Pyodide.
        result = float(numerator) / float(denominator)
        if math.isinf(result):
            warnings.warn("overflow encountered in _safe_divide", RuntimeWarning)
        return result


def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    # TODO: Use loss.fit_intercept_only where appropriate instead of
    # DummyRegressor which is the default given by the `init` parameter,
    # see also _init_state.
    if use_predict_proba:
        # Our parameter validation, set via _fit_context and _parameter_constraints
        # already guarantees that estimator has a predict_proba method.
        predictions = estimator.predict_proba(X)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]  # probability of positive class
        eps = np.finfo(np.float32).eps  # FIXME: This is quite large!
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)


def _update_terminal_regions(
    loss,
    tree,
    X,
    y,
    neg_gradient,
    raw_prediction,
    sample_weight,
    sample_mask,
    learning_rate=0.1,
    k=0,
):
    # compute leaf for each sample in ``X``.
    terminal_regions = tree.apply(X)

    if not isinstance(loss, HalfSquaredError):
        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        if isinstance(loss, HalfBinomialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                # Make a single Newton-Raphson step, see "Additive Logistic Regression:
                # A Statistical View of Boosting" FHT00 and note that we use a slightly
                # different version (factor 2) of "F" with proba=expit(raw_prediction).
                # Our node estimate is given by:
                #    sum(w * (y - prob)) / sum(w * prob * (1 - prob))
                # we take advantage that: y - prob = neg_gradient
                neg_g = neg_gradient.take(indices, axis=0)
                prob = y_ - neg_g
                # numerator = negative gradient = y - prob
                numerator = np.average(neg_g, weights=sw)
                # denominator = hessian = prob * (1 - prob)
                denominator = np.average(prob * (1 - prob), weights=sw)
                return _safe_divide(numerator, denominator)

        elif isinstance(loss, HalfMultinomialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                # we take advantage that: y - prob = neg_gradient
                neg_g = neg_gradient.take(indices, axis=0)
                prob = y_ - neg_g
                K = loss.n_classes
                # numerator = negative gradient * (k - 1) / k
                # Note: The factor (k - 1)/k appears in the original papers "Greedy
                # Function Approximation" by Friedman and "Additive Logistic
                # Regression" by Friedman, Hastie, Tibshirani. This factor is, however,
                # wrong or at least arbitrary as it directly multiplies the
                # learning_rate. We keep it for backward compatibility.
                numerator = np.average(neg_g, weights=sw)
                numerator *= (K - 1) / K
                # denominator = (diagonal) hessian = prob * (1 - prob)
                denominator = np.average(prob * (1 - prob), weights=sw)
                return _safe_divide(numerator, denominator)

        elif isinstance(loss, ExponentialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                neg_g = neg_gradient.take(indices, axis=0)
                # numerator = negative gradient = y * exp(-raw) - (1-y) * exp(raw)
                numerator = np.average(neg_g, weights=sw)
                # denominator = hessian = y * exp(-raw) + (1-y) * exp(raw)
                # if y=0: hessian = exp(raw) = -neg_g
                #    y=1: hessian = exp(-raw) = neg_g
                hessian = neg_g.copy()
                hessian[y_ == 0] *= -1
                denominator = np.average(hessian, weights=sw)
                return _safe_divide(numerator, denominator)

        else:

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                return loss.fit_intercept_only(
                    y_true=y_ - raw_prediction[indices, k],
                    sample_weight=sw,
                )

        # update each leaf (= perform line search)
        for leaf in np.nonzero(tree.children_left == TREE_LEAF)[0]:
            indices = np.nonzero(masked_terminal_regions == leaf)[
                0
            ]  # of terminal regions
            y_ = y.take(indices, axis=0)
            sw = None if sample_weight is None else sample_weight[indices]
            update = compute_update(y_, indices, neg_gradient, raw_prediction, k)

            # TODO: Multiply here by learning rate instead of everywhere else.
            tree.value[leaf, 0, 0] = update

    # update predictions (both in-bag and out-of-bag)
    raw_prediction[:, k] += learning_rate * tree.value[:, 0, 0].take(
        terminal_regions, axis=0
    )


def set_huber_delta(loss, y_true, raw_prediction, sample_weight=None):
    """Calculate and set self.closs.delta based on self.quantile."""
    abserr = np.abs(y_true - raw_prediction.squeeze())
    # sample_weight is always a ndarray, never None.
    delta = _weighted_percentile(abserr, sample_weight, 100 * loss.quantile)
    loss.closs.delta = float(delta)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def custom_loss(param, stacks, preds):
    sigmoid = 1 / (1 + np.exp(-param))

    y_true = np.zeros(preds.shape[0])
    for r, (key, value) in enumerate(stacks.items()):
        X, y, indices, neg_g_view = (
            value[0],
            (value[1]).ravel(),
            value[2],
            value[3],
        )
        y_true[indices] = y

    preds = preds.ravel()
    y_pred = (sigmoid * preds) + ((1 - sigmoid) * preds)

    loss = np.mean((y_true - y_pred) ** 2)
    gradient = -2 * np.dot((y_true - y_pred) * sigmoid * (1 - sigmoid), preds)

    return loss, gradient


def optimize_task_param(stacks, preds, initial_guess):
    result = fmin_l_bfgs_b(custom_loss, x0=initial_guess, args=(stacks, preds))
    optimized_task_param = result[0][0]
    return optimized_task_param


class BaseGB(BaseGradientBoosting):
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
        n_iter_no_change=None,
        tol=1e-4,
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
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    @abstractmethod
    def _encode_y(self, y=None, sample_weight=None):
        """Called by fit to validate and encode y."""

    @abstractmethod
    def _get_loss(self, sample_weight):
        """Get loss object from sklearn._loss.loss."""

    def _gradient_h(self, y, raw_predictions):
        if isinstance(self._loss, HuberLoss):
            set_huber_delta(
                loss=self._loss,
                y_true=y,
                raw_prediction=raw_predictions,
                sample_weight=None,
            )

        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
            sample_weight=None,
        )

        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        return neg_g_view

    def _fit_stage(
        self,
        i,
        stacks,
        raw_predictions,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        sample_weight = None

        updated_stacks = {}
        for key, value in stacks.items():
            y = (value[1]).ravel()
            neg_gradient = self._gradient_h(y, raw_predictions[value[2]])
            updated_value = (*value, neg_gradient)
            updated_stacks[key] = updated_value

        for k in range(self.n_trees_per_iteration_):
            preds = np.zeros_like(raw_predictions)
            trees = np.empty(
                len(
                    updated_stacks,
                ),
                dtype=object,
            )
            for r, (key, value) in enumerate(updated_stacks.items()):
                X, y, indices, neg_g_view = (
                    (value[0]).astype(np.float32),
                    (value[1]).ravel(),
                    value[2],
                    value[3],
                )
                if self._loss.is_multiclass:
                    y = np.array(y == k, dtype=np.float64)

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
                    sample_weight = sample_mask.astype(np.float64)[indices]

                X = X_csc if X_csc is not None else X
                tree.fit(
                    X,
                    neg_g_view[:, k],
                    sample_weight=sample_weight,
                    check_input=False,
                )

                X_for_tree_update = X_csr if X_csr is not None else X
                _update_terminal_regions(
                    self._loss,
                    tree.tree_,
                    X_for_tree_update,
                    y,
                    neg_g_view[:, k],
                    raw_predictions[indices],
                    sample_weight,
                    sample_mask[indices],
                    learning_rate=self.learning_rate,
                    k=k,
                )
                preds[indices] = raw_predictions[indices]

                trees[r] = copy.deepcopy(tree)
            raw_predictions = preds
            optimized_task_param = optimize_task_param(
                updated_stacks,
                raw_predictions,
                np.ones_like(raw_predictions.ravel())
                * np.random.normal(np.mean(y), np.std(y)),
            )

            for r, tree_r in enumerate(trees):
                self.estimators_[i, r, k] = tree_r

        return raw_predictions

    def _set_max_features(self):
        """Set self.max_features_."""
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classifier(self):
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

    def _init_state(self):
        """Initialize model state and allocate model state data structures."""

        self.init_ = self.init
        if self.init_ is None:
            if is_classifier(self):
                self.init_ = DummyClassifier(strategy="prior")
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy="quantile", quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(strategy="quantile", quantile=self.alpha)
            else:
                self.init_ = DummyRegressor(strategy="mean")

        # self.estimators_ = np.empty(
        #     (self.n_estimators, self.n_trees_per_iteration_), dtype=object
        # )

        self.estimators_ = np.empty(
            (self.n_estimators, self.T, self.n_trees_per_iteration_), dtype=object
        )

        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
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
    def fit(self, X, y, task):
        if not self.warm_start:
            self._clear_state()

        unique = np.unique(task)
        self.T = len(unique)
        self.tasks_dic = dict(zip(unique, range(self.T)))

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
        )

        y = self._encode_y(y=y, sample_weight=None)
        y = column_or_1d(y, warn=True)  # TODO: Is this still required?

        self._set_max_features()

        # self.loss is guaranteed to be a string
        self._loss = self._get_loss(sample_weight=None)

        self._init_state()

        stack = pd.DataFrame(np.column_stack((X, y, task)))
        num_features = len(stack.columns) - 2

        def extract_data(group):
            X = group.iloc[:, :num_features].values
            y = group.iloc[:, num_features:-1].values
            indices = group.index.tolist()
            return X, y, indices

        stacks = {}
        for r, group in stack.groupby(stack.columns[-1]):
            stacks[r] = extract_data(group)

        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], self.n_trees_per_iteration_),
                dtype=np.float64,
            )

        else:
            self.init_.fit(X, y)

            raw_predictions = _init_raw_predictions(
                X, self.init_, self._loss, is_classifier(self)
            )

        begin_at_stage = 0

        # The rng state must be preserved if warm_start is True
        self._rng = check_random_state(self.random_state)

        # fit the boosting stages
        n_stages = self._fit_stages(
            X,
            y,
            stacks,
            raw_predictions,
            self._rng,
            begin_at_stage,
        )

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(
        self,
        X,
        y,
        stacks,
        raw_predictions,
        random_state,
        begin_at_stage=0,
    ):
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

        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):
            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                y_oob_masked = y[~sample_mask]
                if i == 0:
                    initial_loss = factor * self._loss(
                        y_true=y_oob_masked,
                        raw_prediction=raw_predictions[~sample_mask],
                        sample_weight=None,
                    )

            raw_predictions = self._fit_stage(
                i,
                stacks,
                raw_predictions,
                sample_mask,
                random_state,
                X_csc=X_csc,
                X_csr=X_csr,
            )

            # track loss
            if do_oob:
                self.train_score_[i] = factor * self._loss(
                    y_true=y[sample_mask],
                    raw_prediction=raw_predictions[sample_mask],
                    sample_weight=None,
                )
                self.oob_scores_[i] = factor * self._loss(
                    y_true=y_oob_masked,
                    raw_prediction=raw_predictions[~sample_mask],
                    sample_weight=None,
                )
                previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = factor * self._loss(
                    y_true=y,
                    raw_prediction=raw_predictions,
                    sample_weight=None,
                )

        return i + 1

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64
            )
        else:
            raw_predictions = _init_raw_predictions(
                X, self.init_, self._loss, is_classifier(self)
            )
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        if check_input:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
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
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

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
