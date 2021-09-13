# -*- coding: utf-8 -*-

"""
ClaSP (Classification Score Profile) Segmentation.

References
----------
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
__all__ = ["ClaSPSegmentation", "find_dominant_window_sizes"]

import numpy as np
import pandas as pd

from queue import PriorityQueue

from sktime.transformations.series.clasp import ClaSPTransformer
from sktime.utils.validation.series import check_series


def find_dominant_window_sizes(TS, offset=0.05):
    """
    Determine the Window-Size using dominant FFT-frequencies.

    Parameters
    ----------
    TS: array
        the time series to determine the periodicity
    offset: float
        Exclusion Radius

    Returns
    -------
    trivial_match: bool
        If the candidate change point is a trivial match
    """

    fourier = np.absolute(np.fft.fft(TS))
    freq = np.fft.fftfreq(TS.shape[0], 1)

    coefs = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            coefs.append(coef)
            window_sizes.append(1 / freq)

    coefs = np.array(coefs)
    window_sizes = np.asarray(window_sizes, dtype=np.int64)

    idx = np.argsort(coefs)[::-1]
    for window_size in window_sizes[idx]:
        if window_size not in range(20, int(TS.shape[0] * offset)):
            continue

        return int(window_size / 2)


def _is_trivial_match(candidate, change_points, n_timepoints, exclusion_radius=0.05):
    """
    Check if a candidate change point is in close proximity to other change points.

    Parameters
    ----------
    candidate: int
        Candidate change point
    change_points: array
        Change points chosen so far
    n_timepoints: int
        Total length
    exclusion_radius: int
        Exclusion Radius for change points

    Returns
    -------
    trivial_match: bool
        If the candidate change point is a trivial match
    """
    change_points = [0] + change_points + [n_timepoints]
    exclusion_radius = np.int64(n_timepoints * exclusion_radius)

    for change_point in change_points:
        left_begin = max(0, change_point - exclusion_radius)
        right_end = min(n_timepoints, change_point + exclusion_radius)
        if candidate in range(left_begin, right_end):
            return True

    return False


def _segmentation(TS, clasp, n_change_points=None, exclusion_radius=0.05):
    """
    Segments the time series by extracting change points.

    Parameters
    ----------
    TS: array
        the time series to be segmented
    clasp:
        the transformer
    n_change_points: int
        the number of change points to find
    exclusion_radius:
        the exclusion zone

    Returns
    -------
    Tuple (array, array, array):
        (predicted_change_points, clasp_profiles, scores)
    """
    period_size = clasp.window_length
    queue = PriorityQueue()

    # compute global clasp
    profile = clasp.transform(TS)
    queue.put(
        (
            -np.max(profile),
            [np.arange(TS.shape[0]).tolist(), np.argmax(profile), profile],
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
                profile = clasp.transform(TS[ranges])
                change_point = np.argmax(profile)
                score = profile[change_point]

                full_profile = np.zeros(len(TS))
                full_profile.fill(0.5)
                np.copyto(
                    full_profile[ranges[0] : ranges[0] + len(profile)],
                    profile.values,
                )

                global_change_point = ranges[0] + change_point

                if not _is_trivial_match(
                    global_change_point,
                    change_points,
                    TS.shape[0],
                    exclusion_radius=exclusion_radius,
                ):
                    queue.put((-score, [ranges, global_change_point, full_profile]))

    return np.array(change_points), np.array(profiles, dtype=object), np.array(scores)


class ClaSPSegmentation(BaseSeriesAnnotator):
    """
    ClaSP (Classification Score Profile) Segmentation.

    Overview:
    ---------
    Using ClaSP for the CPD problem is straightforward: We first compute the profile
    and then choose its global maximum as the change point. The following CPDs
    are obtained using a bespoke recursive split segmentation algorithm.

    Parameters
    ----------
    period_length:         int, default = 10
        size of window for sliding, based on the period length of the data
    n_cps:                 int, default = 1
        the number of change points to search
    fmt :                  str {"dense", "sparse"}, optional (default="sparse")
        Annotation output format:
        * If "sparse", a pd.Series of the found Change Points is returned
        * If "dense", a pd.IndexSeries with the Segmenation of the TS is returned

    Returns
    -------
    pd.Series():
        Found change points

    References
    ----------
    As described in
    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }
    """

    _tags = {"univariate-only": True, "fit-in-predict": True}  # for unit test cases

    def __init__(self, period_length=10, n_cps=1, fmt="sparse"):
        self.period_length = int(period_length)
        self.n_cps = n_cps
        super(ClaSPSegmentation, self).__init__(fmt)

    def _fit(self, X, Y=None):
        # nothing to do
        return True

    def _predict(self, X):
        self.found_cps, self.profiles, self.scores = self._run_clasp(X)

        # Change Points
        if self.fmt == "sparse":
            return pd.Series(self.found_cps)

        # Segmentation
        elif self.fmt == "dense":
            return self._get_interval_series(X, self.found_cps)

    def _predict_scores(self, X):
        self.found_cps, self.profiles, self.scores = self._run_clasp(X)

        # Scores of the Change Points
        if self.fmt == "sparse":
            return pd.Series(self.scores)

        # Full Profile of Segmentation
        # ClaSP creates multiple profiles. Hard to map.
        # Thus, we return the main (first) one
        elif self.fmt == "dense":
            return pd.Series(self.profiles[0])

    def fit_predict(self, X, Y=None):
        """Get shortcut for fit and predict."""
        return self.fit(X).predict(X)

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {"profiles": self.profiles, "scores": self.scores}

    def _run_clasp(self, X):
        X = check_series(X, enforce_univariate=True, allow_numpy=True)

        if isinstance(X, pd.Series):
            X = X.to_numpy()

        clasp_transformer = ClaSPTransformer(window_length=self.period_length).fit(X)

        self.found_cps, self.profiles, self.scores = _segmentation(
            X, clasp_transformer, n_change_points=self.n_cps, exclusion_radius=0.05
        )

        return self.found_cps, self.profiles, self.scores

    def _get_interval_series(self, X, found_cps):
        """Gets the segmentation results based on the found change points.

        Parameters
        ----------
        X:             array
            The time series to be segmented
        found_cps:     array
            Array of change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change pints

        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)
