"""
Information Gain-based Temporal Segmentation.

Information Gain Temporal Segmentation (IGTS) is a method for segmenting
multivariate time series based off reducing the entropy in each segment.

The amount of entropy lost by the segmentations made is called the Information
Gain (IG). The aim is to find the segmentations that have the maximum information
gain for any number of segmentations.

"""

from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import numpy.typing as npt
from attrs import define, asdict
from sortedcontainers import SortedSet
from sktime.base import BaseEstimator


__all__ = ["InformationGainSegmentation"]
__author__ = ["lmmentel"]


@dataclass
class ChangePointResult:
    k: int
    score: float
    change_points: List[int]


def entropy(X: npt.ArrayLike) -> float:
    """Shannons's entropy for time series."""
    p = np.sum(X, axis=0) / np.sum(X)
    p = p[p > 0.0]
    return -np.sum(p * np.log(p))


def generate_segments(X: npt.ArrayLike, change_points: SortedSet) -> npt.ArrayLike:
    """Generate separate segments from time series based on change points."""
    for start, end in zip(change_points[:-1], change_points[1:]):
        yield X[start:end, :]


def generate_segments_pandas(X: npt.ArrayLike, change_points: SortedSet) -> npt.ArrayLike:
    """Generate separate segments from time series based on change points."""
    for interval in pd.IntervalIndex.from_breaks(change_points, closed="both"):
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

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.
    step: : int, default=5
        Step size, or stride for selecting candidate locations of change points.
        Fox example a `step=5` would produce candidates [0, 5, 10, ...]
    verbose: bool, default=False
        If `True` verbose output will be printed

    Notes
    -----
    Based on the work from [1]_.
    - alternative python implementation: https://github.com/cruiseresearchgroup/IGTS-python
    - MATLAB implementation: https://github.com/cruiseresearchgroup/IGTS-matlab
    - paper available at:

    References
    ----------
    .. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
       "Information gain-based metric for recognizing transitions in human activities.",
       Pervasive and Mobile Computing, 38, 92-109, (2017).
       https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081

    """

    # init attributes
    k_max: int = 10
    step: int = 5
    verbose: bool = False

    # computed attributes
    intermediate_results_: List = None

    def identity(self, X: npt.ArrayLike) -> SortedSet:
        """Return identity segmentation, i.e. terminal indexes of the data."""
        return SortedSet([0, X.shape[0]])

    @staticmethod
    def get_candidates(
        n_samples: int, step: int, change_points: SortedSet
    ) -> SortedSet:
        """Generate candidate change points.

        Exclude existing change points.

        TODO: exclude points within a neighborhood of existing change points with neighborhood radius
        """
        return SortedSet(range(0, n_samples, step)).difference(change_points)

    @staticmethod
    def information_gain_cost(X: npt.ArrayLike, change_points: SortedSet) -> float:
        segment_entropies = [
            seg.shape[0] * entropy(seg) for seg in generate_segments(X, change_points)
        ]
        return entropy(X) - sum(segment_entropies) / X.shape[0]

    def find_change_points(self, X: npt.ArrayLike) -> SortedSet:
        """Find change points"""
        n_samples, n_series = X.shape
        ig_max = 0
        self.intermediate_results_ = []

        # by convention
        current_change_points = SortedSet(self.identity(X))

        for k in range(self.k_max):
            # find a point which maximizes score
            for candidate in self.get_candidates(
                n_samples, self.step, current_change_points
            ):
                try_change_points = SortedSet([candidate]).update(current_change_points)
                ig = self.information_gain_cost(X, try_change_points)
                if self.verbose:
                    print(f"{ig=:.5f} for {candidate} current {current_change_points}")
                if ig > ig_max:
                    ig_max = ig
                    best_candidate = candidate

            current_change_points.add(best_candidate)
            self.intermediate_results_.append(
                ChangePointResult(
                    k=k, score=ig_max, change_points=current_change_points
                )
            )
            if self.verbose:
                print(
                    f"BEST {ig_max=:.5f} for {best_candidate} current {current_change_points}"
                )
        return current_change_points


class InformationGainSegmentation(BaseEstimator):
    def __init__(
        self,
        k_max: int = 10,
        step: int = 5,
        verbose: bool = False,
    ):
        self._adaptee_class = IGTS
        self._adaptee = self._adaptee_class(
            k_max=k_max,
            step=step,
            verbose=verbose,
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
        return self._adaptee.find_change_points(X)

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
