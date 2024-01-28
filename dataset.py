import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def toy_data(n_samples, n_classes):
    random_seed = 555
    np.random.seed(random_seed)
    n_features = 2

    X, y = make_classification(
        n_classes=n_classes,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_seed,
    )

    task = np.zeros(X.shape[0], dtype=int)
    df = pd.DataFrame(
        np.column_stack((X, y, task)),
        columns=[f"feature_{i}" for i in range(n_features)] + ["target", "task"],
    )

    noise_data = (np.random.normal(0, 5, X.shape)) + X
    task = np.ones(X.shape[0], dtype=int)
    df_noised = pd.DataFrame(
        np.column_stack((noise_data, y, task)),
        columns=[f"feature_{i}" for i in range(n_features)] + ["target", "task"],
    )
    merged_df = pd.concat([df, df_noised])

    return merged_df
