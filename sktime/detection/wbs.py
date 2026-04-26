"""Wild binary segmentation change point detector."""

from collections import deque

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["priyanshuharshbodhi1"]


class WildBinarySegmentation(BaseDetector):
    """Wild binary segmentation change point detector.

    Extends binary segmentation by searching over random sub-intervals rather
    than always splitting the full segment. This improves detection of nearby
    or weak change points that plain binary segmentation tends to miss.

    Parameters
    ----------
    threshold : float
        Threshold for the CUMSUM statistic. Candidate change points with a
        statistic below this value are discarded.
    n_intervals : int, default=5000
        Number of random sub-intervals drawn at each recursion step.
    min_cp_distance : int, default=0
        Minimum distance between consecutive change points.
    max_iter : int, default=10000
        Maximum recursion iterations before returning accumulated change points.
    random_state : int or None, default=None
        Seed for the random number generator. Pass an integer for reproducibility.

    Notes
    -----
    Based on the wild binary segmentation algorithm described in [1]_. The key
    difference from plain binary segmentation is that M random intervals
    [s_i, e_i] are drawn within the current segment at each step, and the
    candidate change point with the highest CUMSUM statistic across all
    intervals is selected.

    References
    ----------
    .. [1] Fryzlewicz, Piotr. "Wild Binary Segmentation for Multiple Change-Point
           Detection." The Annals of Statistics, vol. 42, no. 6, 2014, pp. 2243-81.
           JSTOR, http://www.jstor.org/stable/43556493. Accessed 4 July 2024.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.wbs import WildBinarySegmentation
    >>> model = WildBinarySegmentation(threshold=1, n_intervals=100, random_state=42)
    >>> X = pd.Series([1, 1, 1, 1, 5, 5, 5, 5])
    >>> model.fit_predict(X)
       ilocs
    0      3
    """

    _tags = {
        "authors": "priyanshuharshbodhi1",
        "maintainers": "priyanshuharshbodhi1",
        "fit_is_empty": True,
        "capability:multivariate": False,
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.Series",
    }

    def __init__(
        self,
        threshold,
        n_intervals=5000,
        min_cp_distance=0,
        max_iter=10000,
        random_state=None,
    ):
        self.threshold = threshold
        self.n_intervals = n_intervals
        self.min_cp_distance = min_cp_distance
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__()

    def _cumsum_statistic(self, X, start, end, change_point):
        """Calculate CUMSUM statistic to evaluate a candidate change point.

        Parameters
        ----------
        X : pd.Series
        start : int
        end : int
        change_point : int

        Returns
        -------
        float
        """
        if change_point < start or change_point >= end:
            raise RuntimeError("The change point must be within 'start' and 'end'.")

        n = end - start + 1
        w_left = np.sqrt((end - change_point) / (n * (change_point - start + 1)))
        w_right = np.sqrt((change_point - start + 1) / (n * (end - change_point)))

        left = X.iloc[start : change_point + 1].to_numpy()
        right = X.iloc[change_point + 1 : end + 1].to_numpy()

        return np.abs(w_left * np.sum(left) - w_right * np.sum(right))

    def _find_change_points(self, X, threshold, min_cp_distance=0, max_iter=10000):
        """Find change points using random sub-interval search.

        Parameters
        ----------
        X : pd.Series
        threshold : float
        min_cp_distance : int
        max_iter : int

        Returns
        -------
        list[int]
        """
        rng = np.random.default_rng(self.random_state)
        change_points = []
        segment_queue = deque([(0, len(X) - 1)])

        i = 0
        while i < max_iter:
            i += 1
            if not segment_queue:
                return change_points

            start, end = segment_queue.popleft()

            cp_start = start + min_cp_distance
            cp_end = end - min_cp_distance

            if cp_end <= cp_start:
                continue

            # draw random sub-intervals within [start, end]
            seg_len = end - start
            if seg_len < 2:
                continue

            s_samples = rng.integers(start, end, size=self.n_intervals)
            e_samples = rng.integers(s_samples + 1, end + 1, size=self.n_intervals)

            best_stat = 0.0
            best_cp = None

            for s, e in zip(s_samples, e_samples):
                local_cp_start = max(s + min_cp_distance, cp_start)
                local_cp_end = min(e - min_cp_distance, cp_end)
                if local_cp_end <= local_cp_start:
                    continue
                for cp in range(local_cp_start, local_cp_end):
                    stat = self._cumsum_statistic(X, s, e, cp)
                    if stat > best_stat:
                        best_stat = stat
                        best_cp = cp

            if best_stat > threshold and best_cp is not None:
                change_points.append(best_cp)
                segment_queue.append((start, best_cp))
                segment_queue.append((best_cp + 1, end))

        return change_points

    def _predict(self, X, Y=None):
        """Find change points in X.

        Parameters
        ----------
        X : pd.Series
        Y : ignored

        Returns
        -------
        pd.Series
            Integer iloc indices of detected change points.
        """
        change_points = self._find_change_points(
            X, self.threshold, self.min_cp_distance, self.max_iter
        )
        change_points.sort()
        return pd.Series(change_points, dtype="int64")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params0 = {"threshold": 1, "n_intervals": 10, "random_state": 0}
        params1 = {
            "threshold": 2,
            "n_intervals": 5,
            "min_cp_distance": 1,
            "random_state": 1,
        }
        return [params0, params1]
