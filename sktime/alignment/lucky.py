# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Lucky sequence alignment."""

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerLuckyDtw(BaseAligner):
    """Alignment path based on lucky dynamic time warping distance.

    This aligner returns the alignment path produced by the lucky time warping
    distance [1]_.
    Uses Euclidean distance for multivariate data.

    Based on code by Krisztian A Buza's research group.

    Parameters
    ----------
    window: int, optional (default=None)
        Maximum distance between indices of aligned series, aka warping window.
        If None, defaults to max(len(ts1), len(ts2)), i.e., no warping window.

    References
    ----------
    ..[1] Stephan Spiegel, Brijnesh-Johannes Jain, and Sahin Albayrak.
        Fast time series classification under lucky time warping distance.
        Proceedings of the 29th Annual ACM Symposium on Applied Computing. 2014.

    Example
    -------
    >>> import pandas as pd
    >>> from sktime.alignment.lucky import AlignerLuckyDtw
    >>> ts1_df = pd.DataFrame({"dim_0": [1, 2, 3, 4, 5]})
    >>> ts2_df = pd.DataFrame({"dim_0": [2, 3, 4, 5, 6]})
    >>> aligner = AlignerLuckyDtw(window=2)
    >>> aligner.fit([ts1_df, ts2_df])
    AlignerLuckyDtw(...)
    >>> alignment = aligner.get_alignment()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "Krisztian A Buza"],
        # estimator type
        # --------------
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": True,  # does compute/return overall distance?
        "capability:distance-matrix": True,  # does compute/return distance matrix?
        "capability:unequal_length": True,  # can align sequences of unequal length?
        "alignment_type": "full",  # does the aligner produce full or partial alignment
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, window=None):
        self.window = window

        super().__init__()

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X
        """
        window = self.window

        ts1, ts2 = X
        ts1 = ts1.values
        ts2 = ts2.values

        len_ts1 = len(ts1)
        len_ts2 = len(ts2)

        if window is None:
            window = max(len_ts1, len_ts2)

        def vec_dist(x):
            return np.linalg.norm(x) ** 2

        d = vec_dist(ts1[0] - ts2[0])

        i = 0
        j = 0
        align_i = [i]
        align_j = [j]

        while i + 1 < len_ts1 or j + 1 < len_ts2:
            d_best = np.inf

            if i + 1 < len_ts1 and j + 1 < len_ts2:
                d_best = vec_dist(ts1[i + 1] - ts2[j + 1])
                new_i = i + 1
                new_j = j + 1

            if i + 1 < len_ts1 and abs(i + 1 - j) <= window:
                d1 = vec_dist(ts1[i + 1] - ts2[j])
                if d1 < d_best:
                    d_best = d1
                    new_i = i + 1
                    new_j = j

            if j + 1 < len_ts2 and abs(j + 1 - i) <= window:
                d2 = vec_dist(ts1[i] - ts2[j + 1])
                if d2 < d_best:
                    d_best = d2
                    new_i = i
                    new_j = j + 1

            d = d + d_best
            i = new_i
            j = new_j
            align_i = align_i + [i]
            align_j = align_j + [j]

        self.align_i_ = align_i
        self.align_j_ = align_j
        self.dist_ = d

        return self

    def _get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        align = pd.DataFrame({"ind0": self.align_i_, "ind1": self.align_j_})
        return align

    def _get_distance(self):
        """Return overall distance of alignment.

            core logic

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)
        Accesses in self:
            Fitted model attributes ending in "_".

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        return self.dist_

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for aligners.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {}
        params1 = {"window": 3}

        return [params0, params1]
