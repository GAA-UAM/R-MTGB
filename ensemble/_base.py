import copy
import warnings
import numpy as np
from tqdm import trange
from numbers import Integral
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from abc import abstractmethod
from scipy.special import expit as sigmoid
from sklearn.utils import check_random_state
from sklearn.base import _fit_context
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._gradient_boosting import (
    _random_sample_mask,
)
from ._utils import _ensemble_pred
from sklearn.tree._tree import DTYPE
from ._losses import CE, MSE
from sklearn.utils.validation import (
    check_random_state,
    _check_sample_weight,
)
from libs._logging import FileHandler, StreamHandler


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
        max_leaf_nodes=2,
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
        self.log_fh = FileHandler()
        self.log_sh = StreamHandler()

        np.random.seed(self.random_state)

    @abstractmethod
    def _encode_y(self, y=None):
        """Called by fit to validate and encode y."""

    @abstractmethod
    def _get_loss(self, sample_weight: np.ndarray):
        """Get loss object from sklearn._loss.loss."""

    def _neg_gradient(
        self, y: np.ndarray, raw_predictions: np.ndarray, task_type: str, i: int
    ):

        neg_gradient = self._loss.negative_gradient(
            y,
            raw_predictions,
        )

        assert np.all(
            np.isfinite(neg_gradient)
        ), "Negative gradient contains NaN or Inf values."
        assert not np.all(
            neg_gradient == 0
        ), "Negative gradient is zero for all samples."
        assert (
            np.min(neg_gradient) >= -1e5 and np.max(neg_gradient) <= 1e5
        ), "Negative gradient values are outside the expected range."

        if self.tasks_dic != None:
            # Multi-task learning
            if task_type == "data_pooling":
                # ∂L/∂ch = (∂L/∂obj()).(∂obj()/∂ch)
                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    neg_gradient[idx_r] *= 1 - self.sigmoid_thetas_[i, r, :]
            else:
                # ∂L/∂r_h= (∂L/∂obj()).(∂obj()/∂rh)
                return neg_gradient
        return neg_gradient

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

        neg_gradient = self._neg_gradient(y, raw_predictions, task_type, i)

        raw_predictions_ = raw_predictions.copy()

        tree = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter="best",
            max_depth=self.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            min_impurity_decrease=0.0,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self._rng,
            ccp_alpha=0.0,
        )

        if self.subsample < 1.0:
            sample_weight = sample_weight * sample_mask.astype(np.float64)

        tree.fit(X, neg_gradient, sample_weight=sample_weight, check_input=False)

        raw_predictions = self._loss.update_terminal_regions(
            tree.tree_,
            X,
            y,
            neg_gradient,
            raw_predictions,
            sample_weight,
            sample_mask,
            self.learning_rate,
        )

        assert np.all(
            np.isfinite(raw_predictions)
        ), f"Raw predictions contain NaN or Inf in stage {i}."
        assert not np.all(
            raw_predictions_ == raw_predictions
        ), f"Raw predictions did not change in stage {i}."

        self.estimators_[i, r] = tree
        if self.tasks_dic is None:
            self.residual_[i, :] = np.abs(neg_gradient).mean(axis=0)
        else:
            self.residual_[i, r, :] = np.abs(neg_gradient).mean(axis=0)

        return raw_predictions

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
            else:
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
                self.init_ = DummyRegressor(strategy="constant", constant=0)
        self.inits_[r,] = copy.deepcopy(self.init_)
        self.inits_[r,].fit(X, y)
        return self._loss.get_init_raw_predictions(X, self.inits_[r,])

    def _init_state(self, y):
        self.estimators_ = np.empty((self.n_estimators, self.T + 1), dtype=object)
        if not self.is_classifier:
            if y.ndim < 2:
                y = y[:, np.newaxis]
            num_cols = y.shape[1]
        elif self.is_classifier:
            num_cols = self._loss.n_class

        # The optimization of theta for each task is performed exclusively in the first block (n_common_estimators).
        self.sigmoid_thetas_ = sigmoid(
            np.zeros(
                (self.n_common_estimators + 1, self.T, self._loss.n_class),
                dtype=np.float64,
            )
        )

        shape = (
            (self.n_estimators, self.T + 1, num_cols)
            if self.tasks_dic is not None
            else (self.n_estimators, num_cols)
        )
        self.residual_ = np.zeros(shape, dtype=np.float32)
        self.train_score_ = np.zeros((self.n_estimators, self.T + 1), dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, "train_score_"):
            del self.train_score_
        if hasattr(self, "init_"):
            del self.init_
            del self.inits_
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

        if self.is_classifier:
            self._loss = CE(len(set(y_train)))

        else:
            self._loss = MSE(1 if y.ndim == 1 else y.shape[1])

        if task_info is not None:
            # Multi-task learning
            X_train, self.t = self._split_task(X_train, task_info)
            unique = np.unique(self.t)
            self.T = len(unique)
            self.tasks_dic = dict(zip(unique, range(self.T)))
            init_common_prediction = self._fit_initial_model(X_train, y_train, 0)
            init_tasks_prediction = np.zeros_like(init_common_prediction)

            for r_label, r in self.tasks_dic.items():
                idx_r = self.t == r_label
                X_r = X_train[idx_r]
                y_r = y_train[idx_r]
                init_tasks_prediction[idx_r] = self._fit_initial_model(X_r, y_r, r + 1)

        else:
            self.T = 0
            self.tasks_dic = None
            init_tasks_prediction = None
            init_common_prediction = self._fit_initial_model(X_train, y_train, 0)

        self._set_max_features()
        self._init_state(y_train)

        self._rng = check_random_state(self.random_state)

        n_stages = self._fit_stages(
            X_train,
            y_train,
            init_common_prediction,
            init_tasks_prediction,
            sample_weight_train,
            sample_weight_val,
            X_val,
            y_val,
            task_info,
        )

        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            self.residual_ = self.residual_[:n_stages]
            self.n_estimators = n_stages

        return self

    def _label_y(self, y: np.ndarray):
        if self.is_classifier:
            if self._loss.n_class != 2:
                y = LabelBinarizer().fit_transform(y)
            else:
                y_transformed = np.zeros((y.shape[0], 2), dtype=np.float64)
                if y.ndim == 2:
                    y = y.squeeze()
                for k in range(2):
                    y_transformed[:, k] = y == k
                y = y_transformed
        return y

    def _subsampling(self, X):
        n_samples = X.shape[0]
        sample_mask = np.ones((n_samples,), dtype=bool)
        if self.subsample < 1.0:
            n_inbag = max(1, int(self.subsample * n_samples))
            sample_mask = _random_sample_mask(n_samples, n_inbag, self._rng)

        return sample_mask

    def _early_stopping(
        self, i, y_val_pred_iter, y_val, sample_weight_val, loss_history
    ):
        """Update loss history and check early stopping criteria."""
        if self.early_stopping is not None:
            val_loss = self._loss(y_val, next(y_val_pred_iter), sample_weight_val)
            loss_history[i % self.early_stopping] = val_loss
            if i >= self.early_stopping and np.all(loss_history <= loss_history[0]):
                print(f"Early stopping at iteration {i}")
                return True
        return False

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
                self.train_score_[i, 0] = self._loss(
                    y=y[sample_mask],
                    raw_predictions=raw_predictions[sample_mask],
                    sample_weight=sample_weight[sample_mask],
                )

            elif self.tasks_dic != None:

                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    sample_mask = self._subsampling(X[idx_r])
                    self.train_score_[i, r + 1] = self._loss(
                        y=y[idx_r][sample_mask],
                        raw_predictions=raw_predictions[idx_r][sample_mask],
                        sample_weight=sample_weight[idx_r][sample_mask],
                    )

        else:
            if self.tasks_dic is None:
                self.train_score_[i, 0] = self._loss(
                    y=y,
                    raw_predictions=raw_predictions,
                    sample_weight=sample_weight,
                )
            elif self.tasks_dic != None:
                for r_label, r in self.tasks_dic.items():
                    idx_r = self.t == r_label
                    self.train_score_[i, r + 1] = self._loss(
                        y=y[idx_r],
                        raw_predictions=raw_predictions[idx_r],
                        sample_weight=sample_weight[idx_r],
                    )

    def _update_prediction(
        self, common_prediction, tasks_prediction, sigmoid_thetas, stage="fit"
    ):
        if stage == "fit":
            t = self.t
            task_dic = self.tasks_dic
        else:
            t = self.t_test
            task_dic = self.tasks_dic_test

        raw_predictions = np.zeros_like(common_prediction)
        for r_label, r in task_dic.items():
            idx_r = t == r_label
            raw_predictions[idx_r] = _ensemble_pred(
                sigmoid_thetas[r], common_prediction[idx_r], tasks_prediction[idx_r]
            )
        return raw_predictions

    def _opt_theta(self, common_prediction, tasks_prediction, y, theta):

        theta = np.atleast_1d(theta)

        grad_theta = self._loss.gradient_theta(
            common_prediction, tasks_prediction, y, theta
        )

        assert np.all(
            np.isfinite(grad_theta)
        ), "Gradient with respect to theta contains NaN or Inf."
        assert not np.all(grad_theta == 0), "Gradient with respect to theta is zero."

        if not self.is_classifier:
            # Finite difference approximation
            epsilon = 1e-3
            theta_plus, theta_minus = theta + epsilon, theta - epsilon
            w_pred_plus = _ensemble_pred(
                sigmoid(theta_plus), common_prediction, tasks_prediction
            )
            w_pred_minus = _ensemble_pred(
                sigmoid(theta_minus), common_prediction, tasks_prediction
            )
            loss_plus, loss_minus = map(
                lambda w: self._loss(y, w, None), [w_pred_plus, w_pred_minus]
            )
            grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
            assert np.allclose(
                grad_theta, grad_approx, rtol=1e-2, atol=1e-2
            ), f"Gradient (w.r.t theta) mismatch detected. Analytic: {np.sum(grad_theta)}, Approx: {grad_approx}"

        return theta - grad_theta * self.learning_rate

    def _task_theta_opt(self, common_prediction, tasks_prediction, y, theta, i):

        theta_out = np.zeros_like(theta, dtype=np.float64)
        for r_label, r in self.tasks_dic.items():
            idx_r = self.t == r_label
            y_r = y[idx_r]

            optimized_theta = self._opt_theta(
                common_prediction[idx_r],
                tasks_prediction[idx_r],
                y_r,
                theta[r],
            )

            self.sigmoid_thetas_[i + 1, r, :] = sigmoid(optimized_theta)
            theta_out[r] = optimized_theta

        return theta_out

    def _fit_stages(
        self,
        X,
        y,
        common_prediction,
        tasks_prediction,
        sample_weight,
        sample_weight_val,
        X_val,
        y_val,
        task_info,
    ) -> np.int32:

        y = self._label_y(y)

        theta = np.zeros(
            (self.T, self._loss.n_class),
            dtype=np.float64,
        )

        # loss_history = None
        # if self.early_stopping is not None:
        #     loss_history = np.full(self.early_stopping, np.inf)
        #     y_val_pred_iter = self._predict(X_val, task_info)
        #     y_val = self._label_y(y_val)

        #     loss_history = np.full(self.early_stopping, np.inf)

        boosting_bar = trange(
            0,
            self.n_estimators,
            leave=True,
            desc="Boosting epochs",
            dynamic_ncols=True,
        )
        if self.tasks_dic != None:
            # Multi-task learning
            for i in boosting_bar:
                if i < self.n_common_estimators:
                    # First training block (Common tasks)

                    common_prediction = self._fit_stage(
                        i,
                        0,
                        X,
                        y,
                        common_prediction,
                        self._subsampling(X),
                        sample_weight,
                        "data_pooling",
                    )

                    # Optimize theta for task alignment
                    theta = self._task_theta_opt(
                        common_prediction, tasks_prediction, y, theta, i
                    )

                    self._track_loss(
                        i,
                        X,
                        y,
                        sample_weight,
                        common_prediction,
                    )

                    # Update ensemble prediction after the last common estimator
                    if i == self.n_common_estimators - 1:
                        ensemble_prediction = self._update_prediction(
                            common_prediction,
                            tasks_prediction,
                            self.sigmoid_thetas_[-1, :, :],
                        )
                # FIXME: Early stopping for multi task learning
                else:
                    # Second training block (Specific tasks)
                    for r_label, r in self.tasks_dic.items():
                        idx_r = self.t == r_label
                        X_r = X[idx_r]
                        y_r = y[idx_r]
                        sample_mask_r = self._subsampling(X_r)
                        prediction = (
                            ensemble_prediction[idx_r]
                            if i == self.n_common_estimators
                            else tasks_prediction[idx_r]
                        )
                        tasks_prediction[idx_r] = self._fit_stage(
                            i,
                            r + 1,
                            X_r,
                            y_r,
                            prediction,
                            sample_mask_r,
                            sample_weight[idx_r],
                            "specific_task",
                        )

                        del prediction
                        del X_r
                        del y_r
                        del sample_mask_r

                    self._track_loss(
                        i,
                        X,
                        y,
                        sample_weight,
                        tasks_prediction,
                    )

                    # FIXME: Early stoppings
                    # if loss_history is not None:
                    #     if self._early_stopping(
                    #         i, y_val_pred_iter, y_val, sample_weight_val, loss_history
                    #     ):

                    #         break

        else:
            for i in boosting_bar:
                common_prediction = self._fit_stage(
                    i,
                    0,
                    X,
                    y,
                    common_prediction,
                    self._subsampling(X),
                    sample_weight,
                )

                self._track_loss(
                    i,
                    X,
                    y,
                    sample_weight,
                    common_prediction,
                )

                # FIXME: Early stopping for single task learning
            ensemble_prediction = common_prediction

        return i + 1

    def _raw_predict_init(self, X: np.ndarray, task_info=None) -> np.ndarray:
        self._check_initialized()
        if (task_info is not None) and (self.tasks_dic is not None):

            self.X_test, self.t_test = self._split_task(X, task_info)
            t = self.t_test
            unique = np.unique(t)
            T_test = len(unique)
            self.tasks_dic_test = dict(zip(unique, range(T_test)))

            # 0_th index (common task estimator)
            self.X_test = self.estimators_[0, 0]._validate_X_predict(
                self.X_test, check_input=True
            )

            init_common_prediction = self._loss.get_init_raw_predictions(
                self.X_test, self.inits_[0]
            )
            init_tasks_prediction = np.zeros_like(init_common_prediction)

            for r_label, r in self.tasks_dic.items():
                if r_label not in self.tasks_dic:
                    raise ValueError(
                        "Task {} not found in the training set".format(r_label)
                    )
                idx_r = t == r_label
                X_r = self.X_test[idx_r]
                X_r = self.estimators_[
                    self.n_common_estimators, int(self.tasks_dic[r_label]) + 1
                ]._validate_X_predict(X_r, check_input=True)

                init_tasks_prediction[idx_r] = self._loss.get_init_raw_predictions(
                    X_r, self.inits_[r + 1]
                )

        elif task_info is None:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
            X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

            init_common_prediction = self._loss.get_init_raw_predictions(
                X, self.inits_[0]
            )

            init_tasks_prediction = None

        return init_common_prediction, init_tasks_prediction

    def _predict(self, X: np.ndarray, task_info=None):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        common_prediction, tasks_prediction = self._raw_predict_init(X, task_info)

        if isinstance(self._loss, MSE) and self._loss.n_class == 1:
            common_prediction = common_prediction.squeeze()
            if task_info:
                tasks_prediction = tasks_prediction.squeeze()

        if self.tasks_dic != None and task_info != None:
            # Multi task learning
            t = self.t_test
            for i, estimator_row in enumerate(self.estimators_):
                if i < self.n_common_estimators:
                    # First inference block (Common tasks)
                    common_prediction += self.learning_rate * estimator_row[0].predict(
                        self.X_test
                    )

                else:
                    # Second inference block (Specific tasks)
                    for r_label, r in self.tasks_dic_test.items():
                        idx_r = t == r_label
                        X_r = self.X_test[idx_r]
                        tasks_prediction[idx_r] += self.learning_rate * estimator_row[
                            r + 1
                        ].predict(X_r)

                        del X_r
                        del idx_r

            ensemble_prediction = self._update_prediction(
                common_prediction,
                tasks_prediction,
                self.sigmoid_thetas_[-1, :, :],
                "inference",
            )

            return ensemble_prediction

        else:
            # Single task learning
            for _, estimator_row in enumerate(self.estimators_):
                common_prediction += self.learning_rate * estimator_row[0].predict(X)
            return common_prediction

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
