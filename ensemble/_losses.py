from abc import abstractmethod
import numpy as np
from sklearn.tree import _tree
from scipy.special import logsumexp
from ._utils import obj as ensemble_pred
from scipy.special import expit as sigmoid
from sklearn.utils.multiclass import type_of_target

TREE_LEAF = _tree.TREE_LEAF


class LossFunction:
    def __init__(self):
        pass

    @abstractmethod
    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate,
    ):
        """Update the terminal regions (=leaves) of the given tree and the prediction"""

    @abstractmethod
    def negative_gradient(self, y, raw_predictions, **kwargs):
        """Compute the negative gradient"""

    @abstractmethod
    def gradient_theta(self, ch, rh, y, theta):
        """Compute the gradient of the loss function with respect to theta"""


class CE(LossFunction):
    """Cross Entropy Loss function designed
    for binary and multiclass classification"""

    def __init__(self, n_classes_):
        super().__init__()
        self.n_classes_ = n_classes_

    def __call__(self, y, raw_predictions, sample_weight=None):

        return np.average(
            -1 * (y * raw_predictions).sum(axis=1) + logsumexp(raw_predictions, axis=1),
            weights=sample_weight,
        )

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate,
    ):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        terminal_regions = tree.apply(X)

        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(
                tree,
                masked_terminal_regions,
                leaf,
                X,
                y,
                residual,
                raw_predictions,
                sample_weight,
            )

        raw_predictions[:, :] += (learning_rate * tree.value[:, :, 0]).take(
            terminal_regions, axis=0
        )

        return raw_predictions

    def negative_gradient(self, y, raw_predictions, **kwargs):

        return y - np.nan_to_num(
            np.exp(raw_predictions - logsumexp(raw_predictions, axis=1, keepdims=True))
        )

    def gradient_theta(self, ch, rh, y, theta):

        # ∂L/∂theta = (∂L/∂obj()).(∂obj()/∂theta)

        sigma = sigmoid(theta)

        dH_dtheta = sigma * (1 - sigma) * ch
        dL_dH = y - np.nan_to_num(
            np.exp(
                ensemble_pred(sigma, ch, rh)
                - logsumexp(ensemble_pred(sigma, ch, rh), axis=1, keepdims=True)
            )
        )

        gradient = dL_dH * dH_dtheta

        return np.sum(gradient, axis=0)

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        n_classes = self.n_classes_
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        sample_weight = sample_weight[:, np.newaxis]
        numerator = np.sum(sample_weight * residual, axis=0)
        numerator *= (n_classes - 1) / n_classes
        denominator = np.sum(
            sample_weight * (y - residual) * (1 - y + residual), axis=0
        )

        epsilon = 1e-5
        tree.value[leaf, :, 0] = np.where(
            abs(denominator) < 1e-150, 0.0, numerator / (denominator + epsilon)
        )

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions


class MSE(LossFunction):
    """Mean Squared Error function designed
    for single and multi-output regression."""

    def __init__(self):
        super().__init__()

    def __call__(self, y, raw_predictions, sample_weight=None):

        if sample_weight is None:
            try:
                init = 0.5 * np.sum((y - raw_predictions.ravel()) ** 2)
            except:
                init = 0.5 * np.sum((y - raw_predictions) ** 2)
        else:
            if y.ndim > 1:
                init = (
                    1
                    / sample_weight.sum()
                    * 0.5
                    * np.sum(sample_weight[:, None] * ((y - raw_predictions) ** 2))
                )
            else:
                init = (
                    1
                    / sample_weight.sum()
                    * 0.5
                    * np.sum(sample_weight * ((y - raw_predictions.ravel()) ** 2))
                )

        return init

    def negative_gradient(self, y, raw_predictions, **kargs):
        if y.ndim > 1:
            neg_gradient = np.squeeze(y) - raw_predictions
        else:
            neg_gradient = np.squeeze(y) - raw_predictions.ravel()

        return neg_gradient

    def gradient_theta(self, ch, rh, y, theta):

        # ∂L/∂theta = (∂L/∂obj()).(∂obj()/∂theta)

        if y.ndim == 1:
            ch, rh = (
                ch.ravel(),
                rh.ravel(),
            )

        sigma = sigmoid(theta)

        dH_dtheta = sigma * (1 - sigma) * ch

        dL_dH = np.squeeze(y) - ensemble_pred(sigma, ch, rh)

        gradient = dL_dH @ dH_dtheta

        assert np.all(np.isfinite(gradient)), "Gradient contains NaN or Inf."
        assert not np.all(gradient == 0), "Gradient is zero for all samples."

        return gradient

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate,
    ):

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if y.ndim > 1:
            for i in range(y.shape[1]):
                raw_predictions[:, i] += learning_rate * tree.predict(X)[:, i, 0]
        else:
            raw_predictions[:, 0] += learning_rate * tree.predict(X).ravel()

        return raw_predictions

    def get_init_raw_predictions(self, X, estimator):
        predictions = estimator.predict(X)
        if (
            type_of_target(predictions) == "continuous-multioutput"
            or "multiclass-multioutput"
        ):
            try:
                predictions = predictions.reshape(-1, predictions.shape[1]).astype(
                    np.float64
                )
            except:
                predictions = predictions.reshape(-1, 1).astype(np.float64)
        else:
            predictions = predictions.reshape(-1, 1).astype(np.float64)
        return predictions
