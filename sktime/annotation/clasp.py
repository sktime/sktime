# -*- coding: utf-8 -*-

"""
ClaSP (Classification Score Profile) Segmentation.

As described in
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
"""

from sktime.annotation.base import BaseSeriesAnnotator

__author__ = ["Arik Ermshaus, Patrick Schäfer"]
__all__ = ["ClaSPSegmentation"]

import numpy as np
import pandas as pd

from queue import PriorityQueue

from sktime.transformations.series.clasp import ClaSPTransformer
from sktime.utils.validation.series import check_series


def _is_trivial_match(candidate, change_points, n_timepoints, exclusion_radius=0.05):
    """
    Check if a candidate change point is in close proximity to other change points.

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


def _segmentation(time_series, clasp, n_change_points=None, offset=0.05):
    """
    Segments the time series by extracting change points.

    :param time_series: the time series to be segmented
    :param clasp: the transformer
    :param n_change_points: the number of change points to find
    :param offset: the exclusion zone
    :return: (predicted_change_points, clasp_profiles, scores)
    """
    period_size = clasp.window_length
    queue = PriorityQueue()

    # compute global clasp
    profile = clasp.transform(time_series)
    queue.put(
        (
            -np.max(profile),
            [np.arange(time_series.shape[0]).tolist(), np.argmax(profile), profile],
        )
    )

    profiles = []
    change_points = []
    scores = []

    for idx in range(n_change_points):
        # should not happen ... safety first
        if queue.empty() is True:
            break

        # get profile with highest change point score
        priority, (profile_range, change_point, full_profile) = queue.get()

        change_points.append(change_point)
        scores.append(-priority)
        profiles.append(full_profile)

        if idx == n_change_points - 1:
            break

        # create left and right local range
        left_range = np.arange(profile_range[0], change_point).tolist()
        right_range = np.arange(change_point, profile_range[-1]).tolist()

        for ranges in [left_range, right_range]:
            # create and enqueue left local profile
            if len(ranges) > period_size:
                profile = clasp.transform(time_series[ranges])
                change_point = np.argmax(profile)
                score = profile[change_point]

                full_profile = np.zeros(len(time_series))
                full_profile.fill(0.5)
                np.copyto(
                    full_profile[ranges[0] : ranges[0] + len(profile)],
                    profile.values,
                )

                global_change_point = ranges[0] + change_point

                if not _is_trivial_match(
                    global_change_point,
                    change_points,
                    time_series.shape[0],
                    exclusion_radius=offset,
                ):
                    queue.put((-score, [ranges, global_change_point, full_profile]))

    return np.array(change_points), np.array(profiles, dtype=object), np.array(scores)


class ClaSPSegmentation(BaseSeriesAnnotator):
    """
    ClaSP (Classification Score Profile) Segmentation.

    As described in
    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }

    Overview:

    Using ClaSP for the CPD problem is straightforward: We first compute the profile
    and then choose its global maximum as the change point. The following CPDs
    are obtained using a bespoke recursive split segmentation algorithm.

    Parameters
    ----------
    period_length:         int, default = 10
        size of window for sliding, based on the period length of the data
    n_cps:                 int, default = 1
        the number of change points to search

    Returns
    -------
    Tuple (numpy.array, numpy.array, numpy.array):
        predicted_change_points, clasp_profiles, scores

    """

    _tags = {"univariate-only": True, "fit-in-predict": True}  # for unit test cases

    def __init__(self, period_length=10, n_cps=1):
        self.period_length = int(period_length)
        self.n_cps = n_cps
        super(ClaSPSegmentation, self).__init__()

    def _fit(self, X, Y=None):
        # nothing to do
        return True

    def _predict(self, X):
        X = check_series(X, enforce_univariate=True, allow_numpy=True)

        if isinstance(X, pd.Series):
            X = X.to_numpy()

        clasp_transformer = ClaSPTransformer(window_length=self.period_length).fit(X)

        self.found_cps, self.profiles, self.scores = _segmentation(
            X, clasp_transformer, n_change_points=self.n_cps, offset=0.05
        )

        return pd.Series(self.found_cps)

    def fit_predict(self, X, Y=None):
        """Get shortcut for fit and predict.

        :param X:
        :param Y:
        :return:
        """
        return self.fit(X).predict(X)

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {"profiles": self.profiles, "scores": self.scores}
