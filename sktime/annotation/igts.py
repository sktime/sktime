# -*- coding: utf-8 -*-
"""
Information Gain-based Temporal Segmentation.

Information Gain Temporal Segmentation (IGTS) is a method for segmenting
multivariate time series based off reducing the entropy in each segment [1]_.

The amount of entropy lost by the segmentations made is called the Information
Gain (IG). The aim is to find the segmentations that have the maximum information
gain for any number of segmentations.

References
----------
.. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
    "Information gain-based metric for recognizing transitions in human activities.",
    Pervasive and Mobile Computing, 38, 92-109, (2017).
    https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081

"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import asdict, define, field

from sktime.base import BaseEstimator

__all__ = ["InformationGainSegmentation"]
__author__ = ["lmmentel"]


@dataclass
class ChangePointResult:
    k: int
    score: float
    change_points: List[int]


def entropy(X: npt.ArrayLike) -> float:
    """Shannons's entropy for time series.

    As defined by equations (3) and (4) from [1]_.

    Parameters
    ----------
    X: array_like
        Time series data as a 2D numpy array with sequence index along rows
        and value series in columns.

    Returns
    -------
    entropy: float
        Computed entropy.
    """
    p = np.sum(X, axis=0) / np.sum(X)
    p = p[p > 0.0]
    return -np.sum(p * np.log(p))


def generate_segments(X: npt.ArrayLike, change_points: List[int]) -> npt.ArrayLike:
    """Generate separate segments from time series based on change points.

    Parameters
    ----------
    X: array_like
        Time series data as a 2D numpy array with sequence index along rows
        and value series in columns.

    change_points: list of int
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.

    Yields
    ------
    segment: npt.ArrayLike
        A segments from the input time series between two consecutive change points
    """
    for start, end in zip(change_points[:-1], change_points[1:]):
        yield X[start:end, :]


def generate_segments_pandas(X: npt.ArrayLike, change_points: List) -> npt.ArrayLike:
    """Generate separate segments from time series based on change points.

    Parameters
    ----------
    X: array_like
        Time series data as a 2D numpy array with sequence index along rows
        and value series in columns.

    change_points: list of int
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.

    Yields
    ------
    segment: npt.ArrayLike
        A segments from the input time series between two consecutive change points
    """
    for interval in pd.IntervalIndex.from_breaks(sorted(change_points), closed="both"):
        yield X[interval.left : interval.right, :]


@define
class IGTS:
    """
    Information Gain based Temporal Segmentation (IGTS).

    IGTS is a n unsupervised method for segmenting multivariate time series
    into non-overlapping segments by locating change points that for which
    the information gain is maximized.

    Information gain (IG) is defined as the amount of entropy lost by the segmentation.
    The aim is to find the segmentation that have the maximum information
    gain for a specified number of segments.

    IGTS uses top-down search method to greedily find the next change point
    location that creates the maximum information gain. Once this is found, it
    repeats the process until it finds `k_max` splits of the time series.

    .. note::

       IGTS does not work very well for univariate series but it can still be
       used if the original univariate series are augmented by an extra feature
       dimensions. A technique proposed in the paper [1]_ us to subtract the
       series from it's largest element and append to the series.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.
    step: : int, default=5
        Step size, or stride for selecting candidate locations of change points.
        Fox example a `step=5` would produce candidates [0, 5, 10, ...]. Has the same
        meaning as `step` in `range` function.

    Attributes
    ----------
    intermediate_results_: list of `ChangePointResult`
        Intermediate segmentation results for each k value, where k=1, 2, ..., k_max

    Notes
    -----
    Based on the work from [1]_.
    - alt. py implementation: https://github.com/cruiseresearchgroup/IGTS-python
    - MATLAB version: https://github.com/cruiseresearchgroup/IGTS-matlab
    - paper available at:

    References
    ----------
    .. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
       "Information gain-based metric for recognizing transitions in human activities.",
       Pervasive and Mobile Computing, 38, 92-109, (2017).
       https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081

    Example
    -------
    >>> from sktime.annotation.datagen import piecewise_normal_multivariate
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> X = piecewise_normal_multivariate(lengths=[10, 10, 10, 10],
    ... means=[[0.0, 1.0], [11.0, 10.0], [5.0, 3.0], [2.0, 2.0]],
    ... variances=0.5)
    >>> X_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    >>> from sktime.annotation.igts import InformationGainSegmentation
    >>> igts = InformationGainSegmentation(k_max=3, step=2)
    >>> y = igts.fit_predict(X_scaled)

    """

    # init attributes
    k_max: int = 10
    step: int = 5

    # computed attributes
    intermediate_results_: List = field(init=False, default=[])

    def identity(self, X: npt.ArrayLike) -> List[int]:
        """Return identity segmentation, i.e. terminal indexes of the data."""
        return sorted([0, X.shape[0]])

    def get_candidates(self, n_samples: int, change_points: List[int]) -> List[int]:
        """Generate candidate change points.

        Also exclude existing change points.

        Parameters
        ----------
        n_samples: int
            Length of the time series.
        change_points: list of ints
            Current set of change points, that will be used to exclude values
            from candidates.

        TODO: exclude points within a neighborhood of existing
        change points with neighborhood radius
        """
        return sorted(
            set(range(0, n_samples, self.step)).difference(set(change_points))
        )

    @staticmethod
    def information_gain_score(X: npt.ArrayLike, change_points: List[int]) -> float:
        """Calculate the information gain score.

        The formula is based on equation (2) from [1]_.

        Parameters
        ----------
        X: array_like
            Time series data as a 2D numpy array with sequence index along rows
            and value series in columns.

        change_points: list of ints
            Locations of change points as integer indexes. By convention change points
            include the identity segmentation, i.e. first and last index + 1 values.

        Returns
        -------
        information_gain: float
            Information gain score for the segmentation corresponding to the change
            points.
        """
        segment_entropies = [
            seg.shape[0] * entropy(seg) for seg in generate_segments(X, change_points)
        ]
        return entropy(X) - sum(segment_entropies) / X.shape[0]

    def find_change_points(self, X: npt.ArrayLike) -> List[int]:
        """Find change points.

        Using a top-down search method, iteratively identify at most
        `k_max` change points that increase the information gain score
        the most.

        Parameters
        ----------
        X: array_like
            Time series data as a 2D numpy array with sequence index along rows
            and value series in columns.

        Returns
        -------
        change_points: list of ints
            Locations of change points as integer indexes. By convention change points
            include the identity segmentation, i.e. first and last index + 1 values.
        """
        n_samples, n_series = X.shape
        if n_series == 1:
            raise ValueError(
                "Detected univariate series, IGTS will not work properly"
                " in this case. Consider augmenting your series to multivariate."
            )
        self.intermediate_results_ = []

        # by convention initialize with the identity segmentation
        current_change_points = self.identity(X)

        for k in range(self.k_max):
            ig_max = 0
            # find a point which maximizes score
            for candidate in self.get_candidates(n_samples, current_change_points):
                try_change_points = {candidate}
                try_change_points.update(current_change_points)
                try_change_points = sorted(try_change_points)
                ig = self.information_gain_score(X, try_change_points)
                if ig > ig_max:
                    ig_max = ig
                    best_candidate = candidate

            current_change_points.append(best_candidate)
            current_change_points.sort()
            self.intermediate_results_.append(
                ChangePointResult(
                    k=k, score=ig_max, change_points=current_change_points.copy()
                )
            )

        return current_change_points


class SegmentationMixin:
    """Mixin with methods useful for segmentation problems."""

    def to_classification(self, change_points: List[int]) -> npt.ArrayLike:
        """Convert change point locations to a classification vector.

        Change point detection results can be treated as classification
        with true change point locations marked with 1's at position of
        the change point and remaining non-change point locations being
        0's.

        For example change points [2, 8] for a time series of length 10
        would result in: [0, 0, 1, 0, 0, 0, 0, 0, 1, 0].
        """
        return np.bincount(change_points[1:-1], minlength=change_points[-1])

    def to_clusters(self, change_points: List[int]) -> npt.ArrayLike:
        """Convert change point locations to a clustering vector.

        Change point detection results can be treated as clustering
        with each segment separated by change points assigned a
        distinct dummy label.

        For example change points [2, 8] for a time series of length 10
        would result in: [0, 0, 1, 1, 1, 1, 1, 1, 2, 2].
        """
        labels = np.zeros(change_points[-1], dtype=np.int32)
        for i, (start, stop) in enumerate(zip(change_points[:-1], change_points[1:])):
            labels[start:stop] = i
        return labels


class InformationGainSegmentation(SegmentationMixin, BaseEstimator):
    """Information Gain based Temporal Segmentation (IGTS) Estimator.

    IGTS is a n unsupervised method for segmenting multivariate time series
    into non-overlapping segments by locating change points that for which
    the information gain is maximized.

    Information gain (IG) is defined as the amount of entropy lost by the segmentation.
    The aim is to find the segmentation that have the maximum information
    gain for a specified number of segments.

    IGTS uses top-down search method to greedily find the next change point
    location that creates the maximum information gain. Once this is found, it
    repeats the process until it finds `k_max` splits of the time series.

    .. note::

       IGTS does not work very well for univariate series but it can still be
       used if the original univariate series are augmented by an extra feature
       dimensions. A technique proposed in the paper [1]_ us to subtract the
       series from it's largest element and append to the series.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.

    step: : int, default=5
        Step size, or stride for selecting candidate locations of change points.
        Fox example a `step=5` would produce candidates [0, 5, 10, ...]. Has the same
        meaning as `step` in `range` function.

    Attributes
    ----------
    change_points_: list of int
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.

    intermediate_results_: list of `ChangePointResult`
        Intermediate segmentation results for each k value, where k=1, 2, ..., k_max

    Notes
    -----
    Based on the work from [1]_.
    - alt. py implementation: https://github.com/cruiseresearchgroup/IGTS-python
    - MATLAB version: https://github.com/cruiseresearchgroup/IGTS-matlab
    - paper available at:

    References
    ----------
    .. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
       "Information gain-based metric for recognizing transitions in human activities.",
       Pervasive and Mobile Computing, 38, 92-109, (2017).
       https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081

    Examples
    --------
    >>> from sktime.annotation.datagen import piecewise_normal_multivariate
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> X = piecewise_normal_multivariate(
    ... lengths=[10, 10, 10, 10],
    ... means=[[0.0, 1.0], [11.0, 10.0], [5.0, 3.0], [2.0, 2.0]],
    ... variances=0.5,
    ... )
    >>> X_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    >>> from sktime.annotation.igts import InformationGainSegmentation
    >>> igts = InformationGainSegmentation(k_max=3, step=2)
    >>> y = igts.fit_predict(X_scaled)

    """

    def __init__(
        self,
        k_max: int = 10,
        step: int = 5,
    ):
        self.k_max = k_max
        self.step = step
        self._adaptee_class = IGTS
        self._adaptee = self._adaptee_class(
            k_max=k_max,
            step=step,
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None):
        """Fit method for compatibility with sklearn-type estimator interface.

        It sets the internal state of the estimator and returns the initialized
        instance.

        Parameters
        ----------
        X: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.

        y: array_like
            Placeholder for compatibility with sklearn-api, not used, default=None.
        """
        return self

    def predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.ArrayLike:
        """Perform segmentation.

        Parameters
        ----------
        X: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.

        y: array_like
            Placeholder for compatibility with sklearn-api, not used, default=None.

        Returns
        -------
        y_pred : array_like
            1D array with predicted segmentation of the same size as the first
            dimension of X. The numerical values represent distinct segments
            labels for each of the data points.
        """
        self.change_points_ = self._adaptee.find_change_points(X)
        self.intermediate_results_ = self._adaptee.intermediate_results_
        return self.to_clusters(self.change_points_)

    def fit_predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.ArrayLike:
        """Perform segmentation.

        A convenience method for compatibility with sklearn-like api.

        Parameters
        ----------
        X: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.

        y: array_like
            Placeholder for compatibility with sklearn-api, not used, default=None.

        Returns
        -------
        y_pred : array_like
            1D array with predicted segmentation of the same size as the first
            dimension of X. The numerical values represent distinct segments
            labels for each of the data points.
        """
        return self.fit(X=X, y=y).predict(X=X, y=y)

    def get_params(self, deep: bool = True) -> Dict:
        """Return initialization parameters.

        Parameters
        ----------
        deep: bool
            Dummy argument for compatibility with sklearn-api, not used.

        Returns
        -------
        params: dict
            Dictionary with the estimator's initialization parameters, with
            keys being argument names and values being argument values.
        """
        return asdict(self._adaptee, filter=lambda attr, value: attr.init is True)

    def set_params(self, **parameters):
        """Set the parameters of this object.

        Parameters
        ----------
        parameters : dict
            Initialization parameters for th estimator.

        Returns
        -------
        self : reference to self (after parameters have been set)
        """
        for key, value in parameters.items():
            setattr(self._adaptee, key, value)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the estimator."""
        return self._adaptee.__repr__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        return {"k_max": 2, "step": 1}
