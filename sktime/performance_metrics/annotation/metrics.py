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


def annotation_error(
    true_change_points: npt.ArrayLike, pred_change_points: npt.ArrayLike
) -> float:
    """
    Annotation error measuring discrepancy in the number of change points.

    Parameters
    ----------
    true_change_points: array_like
        Indexes of true change points
    pred_change_points: array_like
        Indexes of predicted change points

    Returns
    -------
        annotation_error
    """
    true_change_points = check_array(true_change_points, ensure_2d=False)
    pred_change_points = check_array(pred_change_points, ensure_2d=False)
    return abs(true_change_points.size - pred_change_points.size)


def hausdorff_error(
    true_change_points: np.array, pred_change_points: np.array, symmetric: bool = True
) -> float:
    """
    Hausdorff metric measures how far two subsets of a metric space are from each other.

    Parameters
    ----------
        true_change_points: true indices of change points
        pred_change_points: predicted indices of change points
        symmetric: if `True` symmetric Hasudorff distance will be used

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
    true_change_points: npt.ArrayLike, pred_change_points: np.array
) -> float:
    """
    Prediction ratio as the ratio of number of predicted to true change points.

    Parameters
    ----------
    true_change_points: array_like
        Indexes of true change points
    pred_change_points: array_like
        Indexes of predicted change points

    Returns
    -------
        prediction_ratio
    """
    true_change_points = check_array(true_change_points, ensure_2d=False)
    pred_change_points = check_array(pred_change_points, ensure_2d=False)
    return pred_change_points.size / true_change_points.size
