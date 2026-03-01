"""PELT (Pruned Exact Linear Time) change point detector."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["RajdeepKushwaha5"]


class PELT(BaseDetector):
    r"""PELT (Pruned Exact Linear Time) change point detector.

    Implements the PELT algorithm [1]_ for exact, globally optimal detection of
    multiple change points in the mean of a univariate time series.

    PELT minimises the total segmentation cost plus a linear penalty for each
    additional change point:

    .. math::

        \min_{\tau_{1:m}} \left[
            \sum_{i=1}^{m+1} \mathcal{C}(x_{\tau_{i-1}+1:\tau_i})
            + m \cdot \beta
        \right]

    where :math:`\tau_{1:m}` are the change point locations and
    :math:`\mathcal{C}` is the L2 segment cost (negative log-likelihood of a
    Gaussian with unknown mean and known unit variance):

    .. math::

        \mathcal{C}(x_{s:e}) =
            \sum_{t=s}^{e} x_t^2 - \frac{\bigl(\sum_{t=s}^{e} x_t\bigr)^2}{e - s + 1}

    The algorithm is *exact* (finds the globally optimal partition) and runs in
    :math:`O(n)` expected time via a pruning step that discards candidate
    change points that cannot be optimal for any future position.

    Parameters
    ----------
    penalty : float
        Cost penalty :math:`\beta` added for each additional change point.
        Larger values produce fewer, more certain change points.  A popular
        data-driven choice is ``2 * log(n)`` (BIC-like; can be set after
        inspecting the series length).
    min_cp_distance : int, default=2
        Minimum number of observations between two successive change points
        (i.e. minimum segment length).  Must be at least 1.

    Notes
    -----
    The reported change point iloc follows the sktime convention used by
    ``BinarySegmentation``: each value is the *last index of the left
    (pre-change) segment*, i.e. if the new segment begins at position ``s``
    (0-indexed), the change point is reported as ``s - 1``.

    The L2 cost is computed in :math:`O(1)` per segment using prefix sums,
    making the overall algorithm :math:`O(n)` in practice with the PELT
    pruning rule.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal
           detection of changepoints with a linear computational cost.
           Journal of the American Statistical Association, 107(500), 1590-1598.
           https://doi.org/10.1080/01621459.2012.737745

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.pelt import PELT
    >>> X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    >>> model = PELT(penalty=2)
    >>> model.fit_predict(X)
       ilocs
    0     19

    Multiple change points:

    >>> X2 = pd.Series([0.0] * 15 + [5.0] * 15 + [0.0] * 15, dtype=float)
    >>> PELT(penalty=2).fit_predict(X2)
       ilocs
    0     14
    1     29
    """

    _tags = {
        "authors": "RajdeepKushwaha5",
        "maintainers": "RajdeepKushwaha5",
        "fit_is_empty": True,
        "capability:multivariate": False,
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.Series",
    }

    def __init__(self, penalty, min_cp_distance=2):
        if penalty < 0:
            raise ValueError(f"penalty must be non-negative, got penalty={penalty!r}")
        if min_cp_distance < 1:
            raise ValueError(
                f"min_cp_distance must be a positive integer, "
                f"got min_cp_distance={min_cp_distance!r}"
            )
        self.penalty = penalty
        self.min_cp_distance = min_cp_distance
        super().__init__()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prefix_sums(x):
        """Return cumulative sum and sum-of-squares arrays (length n+1).

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Raw float array.

        Returns
        -------
        S : np.ndarray, shape (n+1,)
            ``S[i] = sum(x[:i])``
        S2 : np.ndarray, shape (n+1,)
            ``S2[i] = sum(x[:i]**2)``
        """
        S = np.zeros(len(x) + 1)
        S2 = np.zeros(len(x) + 1)
        S[1:] = np.cumsum(x)
        S2[1:] = np.cumsum(x**2)
        return S, S2

    @staticmethod
    def _segment_cost(S, S2, s, e):
        """Return the L2 cost of segment ``x[s:e]`` (half-open Python slice).

        Cost = sum of squared deviations from segment mean.

        Parameters
        ----------
        S : np.ndarray
            Prefix sums (length n+1).
        S2 : np.ndarray
            Prefix sums of squares (length n+1).
        s : int
            Segment start (inclusive, 0-indexed).
        e : int
            Segment end (exclusive, 0-indexed).

        Returns
        -------
        float
        """
        n_seg = e - s
        if n_seg <= 0:
            return 0.0
        seg_sum = S[e] - S[s]
        seg_sum2 = S2[e] - S2[s]
        return float(seg_sum2 - seg_sum**2 / n_seg)

    def _run_pelt(self, x):
        """Run the core PELT dynamic programme over a raw float array.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Raw float values extracted from the input Series.

        Returns
        -------
        change_points : list of int
            Sorted list of iloc positions (0-indexed) of detected change points,
            where each value is the *last index of the left segment*.
        """
        n = len(x)
        d = self.min_cp_distance
        penalty = self.penalty

        S, S2 = self._build_prefix_sums(x)
        cost = lambda s, e: self._segment_cost(S, S2, s, e)  # noqa: E731

        # F[t] = optimal cost for x[0:t].  F[0] = -penalty so the first
        # segment does not incur an extra penalty (standard initialisation).
        F = np.full(n + 1, np.inf)
        F[0] = -penalty

        # prev[t] = the start of the last segment in the optimal partition of
        # x[0:t].  prev[t] = 0 means the whole x[0:t] is one segment.
        prev = np.full(n + 1, -1, dtype=np.intp)

        # Candidate set: tau values such that the segment x[tau:t] is still
        # potentially optimal.
        candidates = [0]

        for t in range(1, n + 1):
            # Enforce minimum segment length: only tau ≤ t - d are valid starts
            # for the segment ending at t.
            valid = [tau for tau in candidates if tau <= t - d]
            if not valid:
                # No valid start; fall back to the furthest admissible position
                valid = [max(0, t - n)]  # whole series as one segment

            best_cost = np.inf
            best_tau = valid[0]
            for tau in valid:
                c = F[tau] + cost(tau, t) + penalty
                if c < best_cost:
                    best_cost = c
                    best_tau = tau

            F[t] = best_cost
            prev[t] = best_tau

            # PELT pruning: discard tau if it cannot be optimal for any t' > t
            candidates = [tau for tau in candidates if F[tau] + cost(tau, t) <= F[t]]
            # The current position t becomes a candidate start for future segments
            candidates.append(t)

        # Back-track through prev[] to recover change points
        change_points = []
        t = n
        while True:
            s = int(prev[t])
            if s == 0:
                break
            change_points.append(s - 1)  # last index of left segment
            t = s

        change_points.sort()
        return change_points

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def _predict(self, X):
        """Detect change points via PELT.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of int
            Sorted iloc positions of detected change points.  Each value is
            the last index of the left (pre-change) segment, matching the
            convention used by ``BinarySegmentation``.
        """
        x = X.to_numpy(dtype=float)
        change_points = self._run_pelt(x)
        return pd.Series(change_points, dtype="int64")

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
        params : dict or list of dict
            Each dict forms parameters for a valid test instance via ``cls(**params)``.
            ``create_test_instance`` uses the first (or only) dict entry.
        """
        params0 = {"penalty": 2.0}
        params1 = {"penalty": 10.0, "min_cp_distance": 3}
        return [params0, params1]
