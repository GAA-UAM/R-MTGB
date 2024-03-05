import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_circles, make_regression


def _gen_df(X, y, X_with_noise, noise_sample):
    data = np.column_stack((X, y, np.zeros_like(y)))

    data_with_noise = np.column_stack(
        (X_with_noise, y[:noise_sample], np.ones_like(y[:noise_sample]))
    )

    tabular_dataset = np.vstack((data, data_with_noise))
    column_names = [f"feature_{i}" for i in range(X.shape[1])] + ["target", "task"]

    df = pd.DataFrame(tabular_dataset, columns=column_names)

    return df


def _add_noise(X, noise_sample, noise_factor, mean):
    if noise_sample:
        constant_factor = 10
        noise = np.random.normal(mean, noise_factor, size=noise_sample)
        constant_noise = np.full_like(noise, constant_factor)
        scattered_noise = noise + constant_noise
        X = X[:noise_sample, :] + scattered_noise[:, np.newaxis]
        # X = X[:noise_sample, :] + noise[:, np.newaxis]
        return X
    else:
        return X


class toy_clf_dataset:
    def __init__(self, n_samples, noise_sample, seed, noise_factor, mean):
        self.n_samples = n_samples
        self.noise_sample = noise_sample
        self.seed = seed
        self.noise_factor = noise_factor
        self.mean = mean

        np.random.seed(seed)

    def _binary(self):
        X = np.random.rand(self.n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)
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

        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)
        return X, y, X_with_noise

    def _overlapping_data(self, overlap_factor=0.3, noise_factor=0.1):
        X = np.random.rand(self.n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification rule

        # Introduce class overlapping
        overlap_samples = int(overlap_factor * self.n_samples)
        X[:overlap_samples, 1] = X[:overlap_samples, 0] + np.random.normal(
            0, noise_factor, overlap_samples
        )

        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)

        return X, y, X_with_noise

    def _correlated_noise(self, correlation_factor=0.8):
        X = np.random.rand(self.n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        correlated_noise = np.random.normal(
            self.mean, self.noise_factor, size=self.n_samples
        )
        X[:, 1] += correlation_factor * correlated_noise

        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)

        return X, y, X_with_noise

    def _circle(self):
        X, y = make_circles(n_samples=self.n_samples, noise=0.1, factor=0.5)

        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)
        return X, y, X_with_noise

    def __call__(self, data_type):
        dataset_types = {
            "binary": self._binary,
            "multi_class": self._multi_class,
            "overlapping": self._overlapping_data,
            "correlated_noise": self._correlated_noise,
            "circle": self._circle,
        }

        if data_type not in dataset_types:
            raise ValueError(
                f"Invalid data_type. Supported types: {list(dataset_types.keys())}"
            )

        X, y, X_with_noise = dataset_types[data_type]()
        self.noise_sample = X.shape[0] if not self.noise_sample else self.noise_sample
        return _gen_df(X, y, X_with_noise, self.noise_sample)


class toy_reg_dataset:
    def __init__(self, n_samples, noise_sample, seed, noise_factor, mean):
        self.n_samples = n_samples
        self.noise_sample = noise_sample
        self.seed = seed
        self.noise_factor = noise_factor
        self.mean = mean

        np.random.seed(seed)

    def _linear(self, targets=1):
        X, y = make_regression(
            n_samples=self.n_samples, n_features=2, n_targets=targets
        )
        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)
        return X, y, X_with_noise

    def _sinusoidal(self):
        def inverse_sinusoidal_regression(x1, x2, amplitude, frequency, phase, offset):
            return (
                amplitude * np.sin(frequency * x1 + phase)
                + amplitude * np.sin(frequency * x2 + phase)
                + offset
            )

        np.random.seed(42)
        num_points = self.n_samples
        x1_values = np.linspace(0, 2 * np.pi, num_points)
        x2_values = np.linspace(0, 2 * np.pi, num_points)
        amplitude_true = 2.0
        frequency_true = 1.5
        phase_true = np.pi / 3
        offset_true = 1.0
        y_values_true = inverse_sinusoidal_regression(
            x1_values,
            x2_values,
            amplitude_true,
            frequency_true,
            phase_true,
            offset_true,
        )

        noise = np.random.normal(0, 0.5, num_points)
        y_values_noisy = y_values_true + noise
        X = np.column_stack((x1_values, x2_values))
        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)

        return X, y_values_noisy, X_with_noise

    def _full_circle(self):
        angles = np.linspace(0, 2 * np.pi, self.n_samples)

        radius = np.random.uniform(0.5, 1.5, size=self.n_samples)
        x1 = radius * np.cos(angles)
        x2 = radius * np.sin(angles)

        noise = np.random.normal(0, 0.1, size=self.n_samples)
        y = np.sin(angles) + noise

        X = np.column_stack((x1, x2))
        X_with_noise = _add_noise(X, self.noise_sample, self.noise_factor, self.mean)

        return X, y, X_with_noise

    def __call__(self, data_type):
        dataset_types = {
            "linear": self._linear,
            "sinusoidal": self._sinusoidal,
            "full_circle": self._full_circle,
        }

        if data_type not in dataset_types:
            raise ValueError(
                f"Invalid data_type. Supported types: {list(dataset_types.keys())}"
            )

        X, y, X_with_noise = dataset_types[data_type]()
        self.noise_sample = X.shape[0] if not self.noise_sample else self.noise_sample
        return _gen_df(X, y, X_with_noise, self.noise_sample)
