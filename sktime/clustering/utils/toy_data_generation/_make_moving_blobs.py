import numpy as np
from sklearn.datasets import make_blobs

__author__ = ["vagechirkov"]
__all__ = ["make_moving_blobs"]


def make_moving_blobs(
    n_times=3,
    n_samples=15,
    cluster_std=0.10,
    random_state=10,
    centers=np.array([[-1, -1], [0, 0], [1, 1]]),
    movement_mode="constant",
    movement_speed=0.1,
):
    """Generate spatio-temporal data."""
    centers = np.array(centers, dtype=float)
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    for t in range(n_times - 1):
        if movement_mode == "constant":
            centers += movement_speed
        elif movement_mode == "random":
            centers += np.random.uniform(-movement_speed, movement_speed, centers.shape)
        else:
            raise ValueError(f"Unknown movement mode: {movement_mode}")
        _X, _y_true = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        X = np.vstack((X, _X))
        y_true = np.hstack((y_true, _y_true))
    # add time column
    time = np.arange(n_times).repeat(n_samples)
    X = np.column_stack([time, X])
    return X, y_true
