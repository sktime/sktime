# -*- coding: utf-8 -*-
"""
Metrics for evaluating performance of segmentation estimators.

Metrics are suitable for comparing predicted change point sets
against true change points and quantify the error.
"""
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import directed_hausdorff
from sklearn.utils import check_array

__author__ = ["lmmentel"]


def count_error(
    true_change_points: npt.ArrayLike, pred_change_points: npt.ArrayLike
) -> float:
    """
    Error counting the difference in the number of change points.

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
    """
    Compute the Hausdorff distance between two sets of change points.

    .. seealso::

       This function wraps :py:func:`scipy.spatial.distance.directed_hausdorff`

    Parameters
    ----------
    true_change_points: array_like
        Integer indexes (positions) of true change points
    pred_change_points: array_like
        Integer indexes (positions) of predicted change points
    symmetric: bool
        If `True` symmetric Hausdorff distance will be used
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
    """
    Prediction ratio is the ratio of number of predicted to true change points.

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
