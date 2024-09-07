import numpy as np
from sklearn.tree import _tree
from scipy.special import logsumexp
from sklearn.utils.multiclass import type_of_target

TREE_LEAF = _tree.TREE_LEAF


class CE:
    def __init__(self, n_classes_):
        self.K = 1
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
        k=0,
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

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):

        return y - np.nan_to_num(
            np.exp(raw_predictions - logsumexp(raw_predictions, axis=1, keepdims=True))
        )

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


class MSE:

    def __init__(self):
        self.K = 1

    def __call__(self, y, raw_predictions, sample_weight=None):

        if sample_weight is None:
            try:
                init = 0.5 * np.sum((y - raw_predictions.ravel()) ** 2)
            except:
                init = 0.5 * np.sum((y - raw_predictions) ** 2)
        else:
            target_type = type_of_target(y)
            if target_type in ["continuous-multioutput", "multiclass-multioutput"]:
                init = (
                    1
                    / sample_weight.sum()
                    * np.sum(sample_weight[:, None] * ((y - raw_predictions) ** 2))
                )
            elif target_type == "continuous":
                init = (
                    1
                    / sample_weight.sum()
                    * np.sum(sample_weight * ((y - raw_predictions.ravel()) ** 2))
                )

        return init

    def negative_gradient(self, y, raw_predictions, **kargs):
        target_type = type_of_target(y)
        if target_type in ["continuous-multioutput", "multiclass-multioutput"]:
            negative_gradient = raw_predictions - np.squeeze(y)

        elif target_type == "continuous":
            negative_gradient = raw_predictions.ravel() - np.squeeze(y)

        return negative_gradient

    def approx_grad(self, predictions, y):
        epsilon = 1e-3
        num_params = predictions.size
        gradient_approx = np.zeros_like(predictions)

        loss = MSE()
        # Compute numerical gradient for each element
        for i in range(num_params):
            predictions_plus_epsilon = np.copy(predictions)
            predictions_plus_epsilon[i] += epsilon

            loss_original = loss(y, predictions)
            loss_plus_epsilon = loss(y, predictions_plus_epsilon)

            # Approximate gradient
            gradient_approx[i] = (loss_plus_epsilon - loss_original) / epsilon

        return gradient_approx

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        target_type = type_of_target(y)
        if target_type in ["continuous-multioutput", "multiclass-multioutput"]:
            for i in range(y.shape[1]):
                raw_predictions[:, i] += learning_rate * tree.predict(X)[:, i, 0]
        elif target_type == "continuous":
            raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()
        return raw_predictions
