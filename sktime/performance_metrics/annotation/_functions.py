# -*- coding: utf-8 -*-
"""Metrics functions to assess performance on annotation task."""


import numpy as np
from sklearn.metrics.pairwise import paired_euclidean_distances

__author__ = ["ermshaua"]
__all__ = [
    "relative_change_point_distance",
]


def relative_change_point_distance(y_true, y_pred, ts_len):
    """Calculate relative predicted change point error.

    Output is non-negative floating point between 0 to 1. The best value is 0.0.

    Ground truth and predicted change points are matched (using nearest neighbors)
    and mean distance between matches is reported. Matching assumes an equal
    amount of ground truth and predicted change points.

    Parameters
    ----------
    y_true : np.array of shape n_cps
        Ground truth (correct) change points.
    y_pred : np.array of shape n_cps
        Predicted change points.
    ts_len : int
        Time series length.

    Returns
    -------
    cp_distance : float
        Relative CP distance between y_true and y_pred.

    Notes
    -----
    As defined in
    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same amount of CPs.")

    cp_distance = 0
    n_cps = y_true.shape[0]

    for cp_pred in y_pred:
        distances = paired_euclidean_distances(
            np.array([cp_pred] * n_cps).reshape(-1, 1), y_true.reshape(-1, 1)
        )
        cp_true_idx = np.argmin(distances, axis=0)
        cp_true = y_true[cp_true_idx]
        cp_distance += np.abs(cp_pred - cp_true)

    return cp_distance / (n_cps * ts_len)
