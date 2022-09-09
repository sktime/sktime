# -*- coding: utf-8 -*-
"""
Metrics for evaluating performance of annotation estimators.
"""

import numpy.typing as npt
import numpy as np


def check_array(iterable):
    return np.array(iterable)


def annotation_error(true_change_points: npt.ArrayLike, pred_change_points: npt.ArrayLike) -> float:
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
    true_change_points = check_array(true_change_points)
    pred_change_points = check_array(pred_change_points)
    return abs(true_change_points.size - pred_change_points.size)


def prediction_ratio(true_change_points: npt.ArrayLike, pred_change_points: np.array) -> float:
    """
    Compute the prediction ration as the ratio of number of predicted change points
    to true change points.

    Parameters
    ----------
    true_change_points: array_like
        Indexes of true change points
    pred_change_points: array_like 
        Indexes of predicted change points

    Returns
    -------
        prediction_ratio

    References
    ----------
    .. [1]  W-H. Lee, J. Ortiz, B. Ko, R. Lee,
       "Time Segmentation through automatic feature learning", 2018.
    """
    true_change_points = check_array(true_change_points)
    pred_change_points = check_array(pred_change_points)
    return pred_change_points.size / true_change_points.size