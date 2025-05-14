"""ClaSP (Classification Score Profile) Segmentation.

Notes
-----
As described in
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
"""

from sktime.detection.base import BaseDetector

__author__ = ["ermshaua", "patrickzib"]
__all__ = ["ClaSPSegmentation", "find_dominant_window_sizes"]

from queue import PriorityQueue

import numpy as np
import pandas as pd

from sktime.transformations.series.clasp import ClaSPTransformer
from sktime.utils.validation.series import check_series


def find_dominant_window_sizes(X, offset=0.05):
    """Determine the Window-Size using dominant FFT-frequencies.

    Parameters
    ----------
    X : array-like, shape=[n]
        a single univariate time series of length n
    offset : float
        Exclusion Radius

    Returns
    -------
    trivial_match: bool
        If the candidate change point is a trivial match
    """
    fourier = np.absolute(np.fft.fft(X))
    freqs = np.fft.fftfreq(X.shape[0], 1)

    coefs = []
    window_sizes = []

    for coef, freq in zip(fourier, freqs):
        if coef and freq > 0:
            coefs.append(coef)
            window_sizes.append(1 / freq)

    coefs = np.array(coefs)
    window_sizes = np.asarray(window_sizes, dtype=np.int64)

    idx = np.argsort(coefs)[::-1]
    for window_size in window_sizes[idx]:
        if window_size not in range(20, int(X.shape[0] * offset)):
            continue

        return int(window_size / 2)


def _is_trivial_match(candidate, change_points, n_timepoints, exclusion_radius=0.05):
    """Check if a candidate change point is in close proximity to other change points.

    Parameters
    ----------
    candidate : int
        A single candidate change point. Will be chosen if non-trivial match based
        on exclusion_radius.
    change_points : list, dtype=int
        List of change points chosen so far
    n_timepoints : int
        Total length
    exclusion_radius : int
        Exclusion Radius for change points to be non-trivial matches

    Returns
    -------
    trivial_match: bool
        If the 'candidate' change point is a trivial match to the ones in change_points
    """
    change_points = [0] + change_points + [n_timepoints]
    exclusion_radius = np.int64(n_timepoints * exclusion_radius)

    for change_point in change_points:
        left_begin = max(0, change_point - exclusion_radius)
        right_end = min(n_timepoints, change_point + exclusion_radius)
        if candidate in range(left_begin, right_end):
            return True

    return False


def _segmentation(X, clasp, n_change_points=None, exclusion_radius=0.05):
    """Segments the time series by extracting change points.

    Parameters
    ----------
    X : array-like, shape=[n]
        the univariate time series of length n to be segmented
    clasp :
        the transformer
    n_change_points : int
        the number of change points to find
    exclusion_radius :
        the exclusion zone

    Returns
    -------
    Tuple (array-like, array-like, array-like):
        (predicted_change_points, clasp_profiles, scores)
    """
    period_size = clasp.window_length
    queue = PriorityQueue()

    # compute global clasp
    profile = clasp.transform(X)
    queue.put(
        (
            -np.max(profile),
            [np.arange(X.shape[0]).tolist(), np.argmax(profile), profile],
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
                profile = clasp.transform(X[ranges])
                change_point = np.argmax(profile)
                score = profile[change_point]

                full_profile = np.zeros(len(X))
                full_profile.fill(0.5)
                np.copyto(
                    full_profile[ranges[0] : ranges[0] + len(profile)],
                    profile,
                )

                global_change_point = ranges[0] + change_point

                if not _is_trivial_match(
                    global_change_point,
                    change_points,
                    X.shape[0],
                    exclusion_radius=exclusion_radius,
                ):
                    queue.put((-score, [ranges, global_change_point, full_profile]))

    return np.array(change_points), np.array(profiles, dtype=object), np.array(scores)


class ClaSPSegmentation(BaseDetector):
    """ClaSP (Classification Score Profile) Segmentation.

    Using ClaSP for the CPD problem is straightforward: We first compute the profile
    and then choose its global maximum as the change point. The following CPDs
    are obtained using a bespoke recursive split segmentation algorithm.

    Parameters
    ----------
    period_length :         int, default = 10
        size of window for sliding, based on the period length of the data
    n_cps :                 int, default = 1
        the number of change points to search
    exclusion_radius : int
        Exclusion Radius for change points to be non-trivial matches

    Notes
    -----
    As described in
    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }

    Examples
    --------
    >>> from sktime.detection.clasp import ClaSPSegmentation
    >>> from sktime.detection.clasp import find_dominant_window_sizes
    >>> from sktime.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation()
    >>> dominant_period_size = find_dominant_window_sizes(X)
    >>> clasp = ClaSPSegmentation(dominant_period_size, n_cps=1)
    >>> found_cps = clasp.fit_predict(X)
    >>> profiles = clasp.profiles
    >>> scores = clasp.scores
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ermshaua", "patrickzib"],
        "maintainers": "ermshaua",
        # estimator type
        # --------------
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "univariate-only": True,
        "fit_is_empty": True,
        "python_dependencies": "numba",
        "X_inner_mtype": "pd.Series",
    }

    def __init__(self, period_length=10, n_cps=1, exclusion_radius=0.05):
        self.period_length = int(period_length)
        self.n_cps = n_cps
        self.exclusion_radius = exclusion_radius

        super().__init__()

    def _fit(self, X, Y=None):
        """Do nothing, as there is no need to fit a model for ClaSP.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to (time series).
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self : True
        """
        return True

    def _predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series or an IntervalSeries
            Change points in sequence X.
        """
        self.found_cps, self.profiles, self.scores = self._run_clasp(X)
        return pd.Series(self.found_cps)

    def _predict_scores(self, X):
        """Return scores in ClaSP's profile for each annotation.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            Sparse scores for found change points in sequence X.
        """
        self.found_cps, self.profiles, self.scores = self._run_clasp(X)

        # Scores of the Change Points
        scores = pd.Series(self.scores)
        return scores

    def _transform_scores(self, X):
        """Return scores in ClaSP's profile for each annotation.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            Dense scores for found change points in sequence X.
        """
        self.found_cps, self.profiles, self.scores = self._run_clasp(X)

        # ClaSP creates multiple profiles. Hard to map. Thus, we return the main
        # (first) one
        profile = pd.Series(self.profiles[0])
        return profile

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

        clasp_transformer = ClaSPTransformer(
            window_length=self.period_length, exclusion_radius=self.exclusion_radius
        ).fit(X)

        self.found_cps, self.profiles, self.scores = _segmentation(
            X,
            clasp_transformer,
            n_change_points=self.n_cps,
            exclusion_radius=self.exclusion_radius,
        )

        return self.found_cps, self.profiles, self.scores

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
           Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps] The found change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change pints
        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {"period_length": 5, "n_cps": 1}
        params1 = {"period_length": 10, "n_cps": 2, "exclusion_radius": 0.05}
        return [params0, params1]
