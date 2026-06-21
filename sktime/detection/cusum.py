"""CUSUM change point detector."""

import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["RajdeepKushwaha5"]


class CUSUM(BaseDetector):
    r"""CUSUM (cumulative sum) change point detector.

    Implements the two-sided Page CUSUM algorithm [1]_ for offline detection of
    multiple change points in the mean of a univariate time series.

    Two one-sided statistics accumulate deviations from a reference mean ``mu``:

    .. math::

        C_t^+ &= \\max(0,\\, C_{t-1}^+ + X_t - \\mu - k) \\\\
        C_t^- &= \\max(0,\\, C_{t-1}^- - X_t + \\mu - k)

    A change point is declared at iloc ``t - 1`` (last index of the left segment)
    the first time either statistic exceeds ``h``.  After each detection both
    statistics reset to zero and the reference mean is re-estimated from the initial
    samples of the new segment so that subsequent changes can be found.

    Parameters
    ----------
    k : float, default=0.5
        Allowable slack. Set to ``delta / 2`` where ``delta`` is the minimum
        absolute shift to detect. Smaller ``k`` catches smaller shifts but may
        increase false positives.
    h : float, default=5.0
        Detection threshold in the same units as the series. A change is flagged
        when :math:`C_t^+` or :math:`C_t^-` exceeds ``h``. Higher values reduce
        false alarms at the cost of detection lag.
    target : float or None, default=None
        Initial reference mean. If None, estimated from the first ``warmup_len``
        samples of the first segment. After every detection the target is always
        re-estimated from the initial samples of the new segment.
    warmup_len : int or None, default=None
        Number of samples at the start of each segment used to estimate the
        reference mean. Defaults to ``min(max(5, n // 10), 20)`` where ``n`` is
        the series length.

    Notes
    -----
    The reported change point iloc is ``t - 1``, where ``t`` is the alarm index
    (first index where :math:`C_t > h`). This matches the sktime convention used
    in ``BinarySegmentation``: the change point is the last index of the left
    (pre-change) segment.

    After each detected change, the reference mean is re-estimated from the first
    ``warmup_len`` samples of the new segment regardless of whether ``target``
    was explicitly provided; ``target`` only sets the initial baseline.

    References
    ----------
    .. [1] Page, E. S. (1954). Continuous inspection schemes. Biometrika, 41(1/2),
           100-115. https://doi.org/10.2307/2333009

    Examples
    --------
    Detect a single step-up:

    >>> import pandas as pd
    >>> from sktime.detection.cusum import CUSUM
    >>> X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    >>> model = CUSUM(k=0.5, h=4)
    >>> model.fit_predict(X)
       ilocs
    0     19

    Probe the running CUSUM statistic (useful for thresholding):

    >>> X2 = pd.Series([0.0] * 20 + [3.0] * 20, dtype=float)
    >>> scores = model.fit(X2).transform_scores(X2)
    >>> int(scores.argmax())  # index where statistic peaks before the alarm
    20
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

    def __init__(self, k=0.5, h=5.0, target=None, warmup_len=None):
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k!r}")
        if h <= 0:
            raise ValueError(f"h must be positive, got h={h!r}")
        if warmup_len is not None:
            if not isinstance(warmup_len, int) or warmup_len < 1:
                raise ValueError(
                    f"warmup_len must be a positive integer or None, got {warmup_len!r}"
                )
        self.k = k
        self.h = h
        self.target = target
        self.warmup_len = warmup_len
        super().__init__()

    def _effective_warmup(self, n):
        """Return the number of samples used to estimate a segment mean.

        Falls back to a data-driven heuristic when ``warmup_len`` is not set.
        """
        if self.warmup_len is not None:
            return self.warmup_len
        return min(max(5, n // 10), 20)

    def _segment_mean(self, x, start, warmup):
        """Return the mean of up to ``warmup`` samples beginning at ``start``."""
        end = min(start + warmup, len(x))
        return float(x[start:end].mean())

    def _run_cusum(self, x):
        """Run the core two-sided CUSUM loop over a raw float array.

        All three public detection paths (_predict, _predict_scores,
        _transform_scores) call this so the statistics are computed once.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Raw float values extracted from the input Series.

        Returns
        -------
        change_points : list of int
            iloc of each detected change point (``t - 1`` convention).
        alarm_scores : list of float
            Value of ``max(C+, C-)`` at the step that fired each alarm.
        running_scores : list of float
            ``max(C+, C-)`` at every timepoint.  Resets to 0.0 immediately
            after an alarm so that subsequent segments start fresh.
        """
        n = len(x)
        warmup = self._effective_warmup(n)
        mu = (
            self.target if self.target is not None else self._segment_mean(x, 0, warmup)
        )

        change_points = []
        alarm_scores = []
        running_scores = []
        c_pos = 0.0
        c_neg = 0.0

        for t in range(n):
            c_pos = max(0.0, c_pos + x[t] - mu - self.k)
            c_neg = max(0.0, c_neg - x[t] + mu - self.k)
            score = max(c_pos, c_neg)

            if score > self.h and t > 0:
                change_points.append(t - 1)
                alarm_scores.append(score)
                running_scores.append(score)  # record pre-reset value at alarm step
                c_pos = 0.0
                c_neg = 0.0
                mu = self._segment_mean(x, t, warmup)
            else:
                running_scores.append(score)

        return change_points, alarm_scores, running_scores

    def _predict(self, X):
        """Detect change points via two-sided CUSUM.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of int
            iloc positions of detected change points.  Each value is the last
            index of the left (pre-change) segment, matching the convention
            used by ``BinarySegmentation``.
        """
        x = X.to_numpy(dtype=float)
        change_points, _, _ = self._run_cusum(x)
        return pd.Series(change_points, dtype="int64")

    def _predict_scores(self, X):
        """Return sparse CUSUM scores, one per detected change point.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of float
            One score per detected change point, in the same order as the
            change points returned by ``_predict``.  Each value is
            ``max(C+, C-)`` at the step that fired the alarm.
        """
        x = X.to_numpy(dtype=float)
        _, alarm_scores, _ = self._run_cusum(x)
        return pd.Series(alarm_scores, dtype=float)

    def _transform_scores(self, X):
        """Return the running CUSUM statistic at every timepoint.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of float
            ``max(C+, C-)`` at each index of ``X``.  The value at each alarm
            timepoint is the full pre-reset statistic (always above ``h``);
            the corresponding change point (CP) is at the *previous* iloc
            (alarm_t - 1).  From the next timepoint onward the statistic
            resets to 0 and accumulates fresh for the new segment.
        """
        x = X.to_numpy(dtype=float)
        _, _, running_scores = self._run_cusum(x)
        return pd.Series(running_scores, index=X.index, dtype=float)

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
        params0 = {"k": 0.5, "h": 4.0}
        params1 = {"k": 1.0, "h": 8.0, "target": 0.0, "warmup_len": 5}
        return [params0, params1]
