import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

__author__ = ["vagechirkov"]
__all__ = ["make_moving_blobs"]


def make_moving_blobs(
    n_times=3,
    n_samples=15,
    cluster_std=0.10,
    random_state=10,
    centers_origin=np.array([[-1, -1], [0, 0], [1, 1]]),
    movement_mode="constant",
    movement_speed=0.1,
):
    """Generate spatio-temporal data."""
    centers = np.array(centers_origin, dtype=float)
    n_orig_clusters = centers.shape[0]
    centers_ = centers.copy()
    center_time_map = dict.fromkeys(range(n_orig_clusters), 0)
    center_cluster_map = {i: i for i in range(n_orig_clusters)}
    cluster_id = n_orig_clusters
    for t in range(n_times - 1):
        if movement_mode == "constant":
            centers_ += movement_speed
        elif movement_mode == "random":
            # Brownian motion is not interesting for clustering
            # TODO: replace with a levy walk
            centers_ += np.random.uniform(
                -movement_speed, movement_speed, centers_.shape
            )
        else:
            raise ValueError(f"Unknown movement mode: {movement_mode}")
        centers = np.vstack((centers, centers_))

        for i in range(n_orig_clusters):
            center_time_map[cluster_id] = t + 1
            center_cluster_map[cluster_id] = i
            cluster_id += 1

    X, y_true_ = make_blobs(
        n_samples=n_samples * n_times * n_orig_clusters,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    # use center_cluster_map to get true labels
    y_true = np.array([center_cluster_map[i] for i in y_true_])
    time = np.array([center_time_map[i] for i in y_true_])
    X = np.column_stack([time, X])

    # sort by time
    sort_idx = np.argsort(time)
    y_true = y_true[sort_idx]
    X = X[sort_idx]

    X = pd.DataFrame(
        X,
        index=pd.MultiIndex.from_arrays(
            [np.int32(np.arange(X.shape[0])), np.int32(X[:, 0])],
            names=["object_id", "time"],
        ),
        columns=["time", "x", "y"],
    )
    X.drop(columns=["time"], inplace=True)

    return X, y_true
