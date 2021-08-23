#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.annotation.base import BaseSeriesAnnotator

__author__ = ["Arik Ermshaus, Patrick Schäfer"]
__all__ = ["ClaSPSegmentation"]

import numpy as np

from queue import PriorityQueue

from sktime.transformations.series.clasp import ClaSPTransformer
from sktime.utils.validation.series import check_series


def _is_trivial_match(candidate, change_points, n_timepoints, exclusion_radius=0.05):
    """
    Checks if a candidate change point is in close proximity to other change points

    :param candidate:
    :param change_points:
    :param n_timepoints:
    :param exclusion_radius:
    :return:
    """
    change_points = [0] + change_points + [n_timepoints]
    exclusion_radius = np.int64(n_timepoints * exclusion_radius)

    for change_point in change_points:
        left_begin = max(0, change_point - exclusion_radius)
        right_end = min(n_timepoints, change_point + exclusion_radius)
        if candidate in range(left_begin, right_end):
            return True

    return False


def segmentation(time_series, clasp, n_change_points=None, offset=0.05):
    """
    Segments the time series by extracting change points

    :param time_series:
    :param clasp:
    :param n_change_points:
    :param offset:
    :return:
    """

    period_size = clasp.window_length
    queue = PriorityQueue()

    # compute global clasp
    profile = clasp.transform(time_series)
    queue.put(
        (
            -np.max(profile),
            (np.arange(time_series.shape[0]).tolist(), np.argmax(profile)),
        )
    )

    change_points = []
    scores = []

    for idx in range(n_change_points):
        # should not happen ... safety first
        if queue.empty() is True:
            break

        # get profile with highest change point score
        priority, (profile_range, change_point) = queue.get()

        change_points.append(change_point)
        scores.append(-priority)

        if idx == n_change_points - 1:
            break

        # create left and right local range
        left_range = np.arange(profile_range[0], change_point).tolist()
        right_range = np.arange(change_point, profile_range[-1]).tolist()

        # create and enqueue left local profile
        if len(left_range) > period_size:
            left_profile = clasp.transform(time_series[left_range])
            left_change_point = np.argmax(left_profile)
            left_score = left_profile[left_change_point]

            global_change_point = left_range[0] + left_change_point

            if not _is_trivial_match(
                global_change_point,
                change_points,
                time_series.shape[0],
                exclusion_radius=offset,
            ):
                queue.put((-left_score, [left_range, global_change_point]))

        # create and enqueue right local profile
        if len(right_range) > period_size:
            right_profile = clasp.transform(time_series[right_range])
            right_change_point = np.argmax(right_profile)
            right_score = right_profile[right_change_point]

            global_change_point = right_range[0] + right_change_point

            if not _is_trivial_match(
                global_change_point,
                change_points,
                time_series.shape[0],
                exclusion_radius=offset,
            ):
                queue.put((-right_score, [right_range, global_change_point]))

    return profile, np.array(change_points), np.array(scores)


class ClaSPSegmentation(BaseSeriesAnnotator):
    """
    ClaSP (Classification Score Profile) Segmentation,
    as described in

    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }

    Overview:

    Implementation of ClaSP's recursive split segmentation algorithm.

    Parameters
    ----------
    period_length:         int, default = 8
        size of window for sliding, based on the period length of the data
    n_cps:                 int, default = 1
        the number of change points to search

    """

    def __init__(self, period_length=8, n_cps=1):
        self.period_length = period_length
        self.n_cps = n_cps
        super(ClaSPSegmentation, self).__init__()

    def _fit(self, X, Y=None):
        # nothing to do
        return True

    def _predict(self, X):
        X = check_series(X, enforce_univariate=True, allow_numpy=True)

        clasp_transformer = ClaSPTransformer(window_length=self.period_length).fit(X)

        return segmentation(
            X, clasp_transformer, n_change_points=self.n_cps, offset=0.05
        )
