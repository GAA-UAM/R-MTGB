import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_circles


class toy_dataset:
    def __init__(self, n_samples, noise_sample, seed, noise_factor):
        self.n_samples = n_samples
        self.noise_sample = noise_sample
        self.seed = seed
        self.noise_factor = noise_factor

        np.random.seed(seed)

    def _gen_df(self, X, y, X_with_noise):
        data = np.column_stack((X, y, np.zeros_like(y)))

        data_with_noise = np.column_stack(
            (X_with_noise, y[: self.noise_sample], np.ones_like(y[: self.noise_sample]))
        )

        tabular_dataset = np.vstack((data, data_with_noise))
        column_names = [f"feature_{i}" for i in range(X.shape[1])] + ["target", "task"]

        df = pd.DataFrame(tabular_dataset, columns=column_names)

        return df

    def _add_noise(self, X):
        noise = np.random.normal(0, self.noise_factor, size=self.noise_sample)
        X = X[: self.noise_sample, :] + noise[:, np.newaxis]
        return X

    def _binary(self):
        X = np.random.rand(self.n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        X_with_noise = self._add_noise(X)
        return X, y, X_with_noise

    def _multi_class(self):
        X, y = make_classification(
            n_classes=3,
            n_samples=self.n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=self.seed,
        )

        X_with_noise = self._add_noise(X)
        return X, y, X_with_noise

    def _overlapping_data(self, overlap_factor=0.3, noise_factor=0.1):
        X = np.random.rand(self.n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification rule

        # Introduce class overlapping
        overlap_samples = int(overlap_factor * self.n_samples)
        X[:overlap_samples, 1] = X[:overlap_samples, 0] + np.random.normal(
            0, noise_factor, overlap_samples
        )

        X_with_noise = self._add_noise(X)

        return X, y, X_with_noise

    def _correlated_noise(self, correlation_factor=0.8):
        X = np.random.rand(self.n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        correlated_noise = np.random.normal(0, self.noise_factor, size=self.n_samples)
        X[:, 1] += correlation_factor * correlated_noise

        X_with_noise = self._add_noise(X)

        return X, y, X_with_noise

    def _imbalanced_data(self, imbalance_ratio=0.2):

        num_minority_samples = int(self.n_samples * (1 - imbalance_ratio))
        y_minority = np.ones(num_minority_samples)
        y_majority = np.zeros(self.n_samples - num_minority_samples)
        y = np.concatenate((y_minority, y_majority), axis=0)

        X = np.random.rand(self.n_samples, 2)

        # Add noise to minority class
        noise_minority = np.random.normal(
            0, self.noise_factor, size=(num_minority_samples, 2)
        )
        X[:num_minority_samples, :] += noise_minority

        X_with_noise = self._add_noise(X)

        return X, y, X_with_noise

    def _circle(self):
        X, y = make_circles(n_samples=self.n_samples, noise=0.1, factor=0.5)

        X_with_noise = self._add_noise(X)
        return X, y, X_with_noise

    def __call__(self, data_type):
        dataset_types = {
            "binary": self._binary,
            "multi_class": self._multi_class,
            "overlapping": self._overlapping_data,
            "correlated_noise": self._correlated_noise,
            "imbalanced_data": self._imbalanced_data,
            "circle": self._circle,
        }

        if data_type not in dataset_types:
            raise ValueError(
                f"Invalid data_type. Supported types: {list(dataset_types.keys())}"
            )

        X, y, X_with_noise = dataset_types[data_type]()

        return self._gen_df(X, y, X_with_noise)
