import copy
import numpy as np
from ._losses import CE, MSE
from numbers import Integral
from abc import abstractmethod
from ._utils import _ensemble_pred
from sklearn.tree._tree import DTYPE
from sklearn.base import _fit_context
from scipy.special import expit as sigmoid
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble._gradient_boosting import (
    _random_sample_mask,
)
from sklearn.utils.validation import (
    check_random_state,
    _check_sample_weight,
    check_is_fitted, validate_data
)


class BaseRMTGB(BaseGradientBoosting):
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_iter_1st,
        n_iter_2nd,
        n_iter_3rd,
        criterion,
        max_depth,
        init,
        subsample,
        max_features,
        random_state,
        validation_fraction=0.1,
        early_stopping=None,
    ):
        self.n_iter_1st = n_iter_1st
        self.n_iter_2nd = n_iter_2nd
        self.n_iter_3rd = n_iter_3rd
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.init = init
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.is_classifier = False
        self.early_stopping = early_stopping

    @abstractmethod
    def _encode_y(self, y=None):
        """Called by fit to validate and encode y."""

    @abstractmethod
    def _get_loss(self, sample_weight: np.ndarray):
        """Get loss object from sklearn._loss.loss."""

    def _neg_gradient(
        self,
        neg_gradient,
        i: int,
        key: str,
    ):

        if key == 2:  # Non-outlier function
            for r_label, r in self.tasks_dic.items():
                idx_r = self.t == r_label
                neg_gradient[idx_r] = neg_gradient[idx_r] * (1 - sigmoid(self.theta[r]))
        elif key == 1:  # outlier function
            for r_label, r in self.tasks_dic.items():
                idx_r = self.t == r_label
                neg_gradient[idx_r] = neg_gradient[idx_r] * sigmoid(self.theta[r])

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
        key=0,
    ):

        if self.subsample < 1.0:
            sample_weight *= sample_mask.astype(np.float64)

        neg_gradient = self._loss.negative_gradient(
            y,
            raw_predictions,
        )

        if key > 0:
            # Handle second training block (outlier-aware)
            neg_gradient = self._neg_gradient(
                neg_gradient,
                i,
                key,
            )

        raw_predictions, self.estimators_[i, r, key] = self._fit_tree(
            raw_predictions,
            neg_gradient,
            X,
            y,
            sample_weight,
            sample_mask,
        )

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

    def _fit_initial_model(self, X, y):
        if not hasattr(self, "init_"):
            self.init_ = self.init
        else:
            del self.init_
            self.init_ = self.init
        if self.init_ is None:
            if self.is_classifier:
                self.init_ = DummyClassifier(strategy="constant", constant=0)
            else:
                self.init_ = DummyRegressor(strategy="constant", constant=0)
        self.init_.fit(X, y)
        return self._loss.get_init_raw_predictions(X, self.init_)

    def _init_state(self, y):
        self.total_iter = int(
            np.sum([self.n_iter_1st, self.n_iter_2nd, self.n_iter_3rd])
        )
        n_training_blocks = 3
        self.estimators_ = np.empty(
            (self.total_iter, self.T, n_training_blocks), dtype=object
        )
        if not self.is_classifier:
            if y.ndim < 2:
                y = y[:, np.newaxis]

        self.train_score_ = np.zeros((self.total_iter, self.T + 1, 3), dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty((0, 0, 0), dtype=object)
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
    def _split_task(self, X: np.ndarray) -> np.ndarray:

        # Ensuring that every distinct value is replaced with an index, beginning from zero.
        unique_values = np.unique(X[:, self.task_info])
        mapping = {value: index for index, value in enumerate(unique_values)}
        X[:, self.task_info] = np.vectorize(mapping.get)(X[:, self.task_info])

        # Separating the input array and task indices.
        X_task = X[:, self.task_info]
        X_data = np.delete(X, self.task_info, axis=1).astype(float)
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

    def fit(self, X: np.ndarray, y: np.ndarray, task_info=-1):

        if not task_info:
            task_info = -1
        self.task_info = task_info

        self._clear_state()

        sample_weight = _check_sample_weight(None, X)

        X, y = validate_data(
            self,
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

        X_train, self.t = self._split_task(X_train)
        unique = np.unique(self.t)
        self.T = len(unique)
        self.tasks_dic = dict(zip(unique, range(self.T)))
        init_prediction = self._fit_initial_model(X_train, y_train)

        self._set_max_features()
        self._init_state(y_train)

        self._rng = check_random_state(self.random_state)

        n_stages = self._fit_stages(
            X_train,
            y_train,
            init_prediction,
            sample_weight_train,
            sample_weight_val,
            X_val,
            y_val,
        )

        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            # self.total_iter = n_stages

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

    def _update_prediction(
        self,
        meta_prediction,
        common_prediction_outlier,
        common_prediction_nonoutlier,
        tasks_prediction,
        sigmoid_thetas,
        stage="fit",
    ):
        if stage == "fit":
            t = self.t
            task_dic = self.tasks_dic
        else:
            t = self.t_test
            task_dic = self.tasks_dic_test

        raw_predictions = np.zeros_like(meta_prediction)
        for r_label, r in task_dic.items():
            idx_r = t == r_label
            raw_predictions[idx_r] = _ensemble_pred(
                sigmoid_thetas[r],
                meta_prediction[idx_r],
                common_prediction_outlier[idx_r],
                common_prediction_nonoutlier[idx_r],
                tasks_prediction[idx_r],
            )
        return raw_predictions

    def _opt_theta(
        self,
        p_meta,
        p_out,
        p_non_out,
        p_task,
        y,
        theta,
    ):

        grad_theta = self._loss.gradient_theta(
            p_meta,
            p_out,
            p_non_out,
            p_task,
            y,
            theta,
        )

        return theta - (self.learning_rate * grad_theta)

    def _opt_theta_per_task(
        self,
        p_meta,
        p_out,
        p_non_out,
        p_task,
        y,
        theta,
    ):

        theta_out = np.zeros_like(theta, dtype=np.float64)
        for r_label, r in self.tasks_dic.items():
            idx_r = self.t == r_label
            y_r = y[idx_r]

            optimized_theta = self._opt_theta(
                p_meta[idx_r],
                p_out[idx_r],
                p_non_out[idx_r],
                p_task[idx_r],
                y_r,
                theta[r],
            )

            theta_out[r] = optimized_theta

        return theta_out

    def _fit_tree(
        self, raw_predictions, neg_gradient, X, y, sample_weight, sample_mask
    ):

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

        tree.fit(
            X,
            neg_gradient,
            sample_weight=sample_weight,
            check_input=False,
        )

        copy_raw_predictions = raw_predictions.copy()

        raw_prediction_out = self._loss.update_terminal_regions(
            tree.tree_,
            X,
            y,
            neg_gradient,
            copy_raw_predictions,
            sample_weight,
            sample_mask,
            self.learning_rate,
        )

        return raw_prediction_out, copy.copy(tree)

    def _fit_stages(
        self,
        X,
        y,
        init_prediction,
        sample_weight,
        sample_weight_val,
        X_val,
        y_val,
    ) -> np.int32:

        y = self._label_y(y)

        self.theta = np.random.randn(self.T) * 0.1

        p_meta = init_prediction
        p_out = p_meta * 0.0
        p_non_out = p_meta * 0.0
        p_task = p_meta * 0.0

        for i in range(self.n_iter_1st):
            x_subsample = self._subsampling(X)
            if i == 0:
                p_meta = self._update_prediction(
                    p_meta,
                    p_out,
                    p_non_out,
                    p_task,
                    sigmoid(self.theta),
                )

            p_meta = self._fit_stage(
                i,
                0,
                X,
                y,
                p_meta,
                x_subsample,
                sample_weight,
                0,
            )

            # self._track_loss(i, X, y, sample_weight, p_meta)
        for cont in range(self.n_iter_2nd):
            j = cont + self.n_iter_1st
            x_subsample = self._subsampling(X)
            ensemble_prediction = self._update_prediction(
                p_meta,
                p_out,
                p_non_out,
                p_task,
                sigmoid(self.theta),
            )

            new_ensemble_prediction = self._fit_stage(
                j,
                0,
                X,
                y,
                ensemble_prediction,
                x_subsample,
                sample_weight,
                1,
            )

            out_tree_cont = (
                new_ensemble_prediction - ensemble_prediction
            )  # Total hack to separate ensemble_output from outlier task output
            p_out_old = p_out.copy()
            p_out += out_tree_cont

            new_ensemble_prediction = self._fit_stage(
                j,
                0,
                X,
                y,
                ensemble_prediction,
                x_subsample,
                sample_weight,
                2,
            )

            non_out_tree_cont = (
                new_ensemble_prediction - ensemble_prediction
            )  # Total hack to separate ensemble_output from non outlier task output
            p_non_out_old = p_non_out.copy()
            p_non_out += non_out_tree_cont

            # Optimize theta values per task
            self.theta = self._opt_theta_per_task(
                p_meta,
                p_out_old,
                p_non_out_old,
                p_task,
                y,
                self.theta,
            )

        for cont in range(self.n_iter_3rd):
            k = cont + self.n_iter_1st + self.n_iter_2nd
            x_subsample = self._subsampling(X)
            ensemble_prediction = self._update_prediction(
                p_meta,
                p_out,
                p_non_out,
                p_task,
                sigmoid(self.theta),
            )

            new_ensemble_prediction = ensemble_prediction.copy()

            for r_label, r in self.tasks_dic.items():
                idx_r = self.t == r_label
                X_r, y_r, sample_weight_r, x_subsample_r = (
                    X[idx_r],
                    y[idx_r],
                    sample_weight[idx_r],
                    x_subsample[idx_r],
                )
                # Update task-specific predictions
                new_ensemble_prediction[idx_r] = self._fit_stage(
                    k,
                    r,
                    X_r,
                    y_r,
                    ensemble_prediction[idx_r],
                    x_subsample_r,
                    sample_weight_r,
                    0,
                )

                # DHL total hack to separate ensemble_output from specitific task output
                p_task[idx_r] += (
                    new_ensemble_prediction[idx_r] - ensemble_prediction[idx_r]
                )

                del X_r
                del y_r
                del sample_weight_r
                del x_subsample_r


        return self.n_iter_1st + self.n_iter_2nd + self.n_iter_3rd

    def _raw_predict_init(self, X: np.ndarray) -> np.ndarray:
        self._check_initialized()

        self.X_test, self.t_test = self._split_task(X)
        t = self.t_test
        unique = np.unique(t)
        T_test = len(unique)
        self.tasks_dic_test = dict(zip(unique, range(T_test)))

        # 0_th index (meta task estimator)
        if self.estimators_[0, 0, 0] is None:
            # We only arrive here if there is no first stage and there is second stage
            self.X_test_meta = self.estimators_[0, 0, 1]._validate_X_predict(
                self.X_test, check_input=True
            )
        else:
            self.X_test_meta = self.estimators_[0, 0, 0]._validate_X_predict(
                self.X_test, check_input=True
            )

        p_meta = self._loss.get_init_raw_predictions(self.X_test_meta, self.init_)

        # Make sure everything is zero except the initial prediction
        p_out = p_meta.copy() * 0.0
        p_non_out = p_meta.copy() * 0.0
        p_task = p_meta.copy() * 0.0

        return (
            p_meta,
            p_out,
            p_non_out,
            p_task,
        )

    def _predict(self, X: np.ndarray):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        (
            p_meta,
            p_out,
            p_non_out,
            p_task,
        ) = self._raw_predict_init(X)

        if isinstance(self._loss, MSE) and self._loss.n_class == 1:
            p_meta = p_meta.squeeze()
            p_out = p_out.squeeze()
            p_non_out = p_non_out.squeeze()
            p_task = p_task.squeeze()

        t = self.t_test

        for _, estimator_row in enumerate(self.estimators_[: self.n_iter_1st]):
            # First inference block (Meta learning)
            p_meta += self.learning_rate * estimator_row[0][0].predict(self.X_test_meta)

        for _, estimator_row in enumerate(
            self.estimators_[self.n_iter_1st : (self.n_iter_1st + self.n_iter_2nd)]
        ):
            # Second inference block (Common task learning)
            p_out += self.learning_rate * estimator_row[0][1].predict(self.X_test_meta)

            p_non_out += self.learning_rate * estimator_row[0][2].predict(
                self.X_test_meta
            )
        for _, estimator_row in enumerate(
            self.estimators_[(self.n_iter_1st + self.n_iter_2nd) :]
        ):
            # Third inference block (tasks-specific learning)
            for r_label, r in self.tasks_dic_test.items():
                idx_r = t == r_label
                X_r = self.X_test_meta[idx_r]
                p_task[idx_r] += self.learning_rate * estimator_row[r][0].predict(X_r)

                del X_r
                del idx_r

        ensemble_prediction = self._update_prediction(
            p_meta,
            p_out,
            p_non_out,
            p_task,
            sigmoid(self.theta),
            "inference",
        )

        return ensemble_prediction
