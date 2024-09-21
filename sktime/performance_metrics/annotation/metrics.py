"""Metrics for evaluating performance of segmentation estimators.

Metrics are suitable for comparing predicted change point sets against true change
points and quantify the error.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from sklearn.utils import check_array

__author__ = ["lmmentel"]


def count_error(
    true_change_points: npt.ArrayLike, pred_change_points: npt.ArrayLike
) -> float:
    """Error counting the difference in the number of change points.

    Parameters
    ----------
    true_change_points: array_like
        Integer indexes (positions) of true change points
    pred_change_points: array_like
        Integer indexes (positions) of predicted change points

    Returns
    -------
        count_error
    """
    true_change_points = check_array(true_change_points, ensure_2d=False)
    pred_change_points = check_array(pred_change_points, ensure_2d=False)
    return abs(true_change_points.size - pred_change_points.size)


def hausdorff_error(
    true_change_points: npt.ArrayLike,
    pred_change_points: npt.ArrayLike,
    symmetric: bool = True,
    seed: int = 0,
) -> float:
    """Compute the Hausdorff distance between two sets of change points.

    .. seealso::

       This function wraps :py:func:`scipy.spatial.distance.directed_hausdorff`

    Parameters
    ----------
    true_change_points: array_like
        Integer indexes (positions) of true change points
    pred_change_points: array_like
        Integer indexes (positions) of predicted change points
    symmetric: bool
        If ``True`` symmetric Hausdorff distance will be used
    seed: int, default=0
        Local numpy.random.RandomState seed. Default is 0, a random
        shuffling of u and v that guarantees reproducibility.

    Returns
    -------
        Hausdorff error.
    """
    a = np.array(true_change_points).reshape(-1, 1)
    b = np.array(pred_change_points).reshape(-1, 1)

    d = directed_hausdorff(a, b)[0]

    if symmetric:
        d = max(d, directed_hausdorff(b, a)[0])

    return d


def prediction_ratio(
    true_change_points: npt.ArrayLike, pred_change_points: npt.ArrayLike
) -> float:
    """Prediction ratio is the ratio of number of predicted to true change points.

    Parameters
    ----------
    true_change_points: array_like
        Integer indexes (positions) of true change points
    pred_change_points: array_like
        Integer indexes (positions) of predicted change points

    Returns
    -------
        prediction_ratio
    """
    true_change_points = check_array(true_change_points, ensure_2d=False)
    pred_change_points = check_array(pred_change_points, ensure_2d=False)
    return pred_change_points.size / true_change_points.size


def padded_f1(true_change_points, pred_change_points, pad):
    """Calculate padded F1 score for change point detection.

    Parameters
    ----------
    true_change_points: pd.Series
        True change point positions. Can be integers, floats or datetimes.
    precicted_change_points: pd.Series
        Precicted change point positions. Can be integers, floats or datetimes.
    pad: int, float, timdelta
        Used to pad the true change points. If a predicted change point falls within
        the range of the padded change then then change point has been correctly
        identified.

    Returns
    -------
    float
        Padded f1 score

    References
    ----------
    .. [1] Gerrit J. J. van den Burg and Christopher K. I. Williams, An Evaluation of
           Change Point Detection Algorithms, 2022, https://arxiv.org/abs/2003.06222
    """
    true_change_points = pd.Series(true_change_points)
    pred_change_points = pd.Series(pred_change_points)

    boundary_left = true_change_points - pad
    boundary_right = true_change_points + pad
    true_cp_intervals = pd.IntervalIndex.from_arrays(boundary_left, boundary_right)

    false_positives = 0
    tp_and_fn = pd.Series(False, index=true_cp_intervals)

    for cp in pred_change_points:
        boolean_mask = tp_and_fn.index.contains(cp)
        if not boolean_mask.any():
            false_positives += 1
        else:
            tp_and_fn = tp_and_fn | boolean_mask

    true_positives = tp_and_fn.sum()
    false_negatives = (~tp_and_fn).sum()

    # Avoid division by zero to mimic sklearn behaviour
    denom = 2 * true_positives + false_positives + false_negatives
    if denom == 0:
        return 0.0

    padded_f1 = 2 * true_positives / denom
    return padded_f1
