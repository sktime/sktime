"""EWMA (Exponentially Weighted Moving Average) control chart detector."""

import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["RajdeepKushwaha5"]


class EWMA(BaseDetector):
    r"""EWMA (Exponentially Weighted Moving Average) control chart detector.

    Implements the EWMA control chart [1]_ for offline detection of multiple
    change points in the mean of a univariate time series.

    The EWMA statistic smooths the series with exponential weights:

    .. math::

        Z_t = \lambda X_t + (1 - \lambda) Z_{t-1}, \qquad Z_0 = \mu_0

    A change point is declared at iloc ``t - 1`` (last index of the left
    segment) the first time the deviation from the reference mean exceeds the
    threshold ``h``:

    .. math::

        |Z_t - \mu_0| > h

    After each detection, ``Z`` is re-initialised to the reference mean of the
    new segment, so that subsequent changes can be found.

    EWMA complements CUSUM: with a small ``lambda`` it detects *small sustained
    shifts* earlier than CUSUM because the smoothing reduces the variance of the
    statistic and the effective control limit narrows.

    Parameters
    ----------
    lam : float, default=0.2
        Smoothing factor (``0 < lam <= 1``).  ``lam = 1`` reduces to a raw
        threshold on each individual observation.  Smaller values produce a
        smoother statistic that reacts slowly to sudden large shifts but is
        more sensitive to small, persistent shifts.
    h : float, default=1.0
        Detection threshold.  A change is flagged when
        :math:`|Z_t - \mu_0| > h`.  Higher values reduce false alarms at the
        cost of detection lag.
    target : float or None, default=None
        Initial reference mean :math:`\mu_0`.  If ``None``, estimated from
        the first ``warmup_len`` samples of the current segment.
    warmup_len : int or None, default=None
        Number of samples used to estimate the reference mean of each segment.
        Defaults to ``min(max(5, n // 10), 20)`` where ``n`` is the series
        length.

    Notes
    -----
    The reported change point iloc is ``t - 1``, where ``t`` is the alarm
    index (first index where :math:`|Z_t - \mu| > h`).  This matches the
    sktime convention used in ``BinarySegmentation`` and ``CUSUM``.

    References
    ----------
    .. [1] Roberts, S. W. (1959). Control chart tests based on geometric moving
           averages. Technometrics, 1(3), 239-250.
           https://doi.org/10.1080/00401706.1959.10489860

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.ewma import EWMA
    >>> X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    >>> model = EWMA(lam=0.2, h=1.0)
    >>> model.fit_predict(X)
       ilocs
    0     20

    Two change points — up then back down:

    >>> X2 = pd.Series([0.0] * 15 + [5.0] * 15 + [0.0] * 15, dtype=float)
    >>> EWMA(lam=0.2, h=1.0).fit_predict(X2)
       ilocs
    0     15
    1     30
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

    def __init__(self, lam=0.2, h=1.0, target=None, warmup_len=None):
        if not (0 < lam <= 1):
            raise ValueError(f"lam must satisfy 0 < lam <= 1, got lam={lam!r}")
        if h <= 0:
            raise ValueError(f"h must be positive, got h={h!r}")
        if warmup_len is not None and warmup_len < 1:
            raise ValueError(
                f"warmup_len must be a positive integer or None, got {warmup_len!r}"
            )
        self.lam = lam
        self.h = h
        self.target = target
        self.warmup_len = warmup_len
        super().__init__()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_warmup(self, n):
        """Return the number of samples used to estimate a segment mean."""
        if self.warmup_len is not None:
            return self.warmup_len
        return min(max(5, n // 10), 20)

    @staticmethod
    def _segment_mean(x, start, warmup):
        """Return the mean of up to ``warmup`` samples beginning at ``start``."""
        end = min(start + warmup, len(x))
        return float(x[start:end].mean())

    def _run_ewma(self, x):
        """Run the core EWMA loop over a raw float array.

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
            Value of ``|Z_t - mu|`` at the step that fired each alarm.
        running_scores : list of float
            ``|Z_t - mu|`` at every timepoint.  Resets immediately after
            each alarm so that subsequent segments start fresh.
        """
        n = len(x)
        warmup = self._effective_warmup(n)
        mu = (
            self.target if self.target is not None else self._segment_mean(x, 0, warmup)
        )
        lam = self.lam
        h = self.h

        change_points = []
        alarm_scores = []
        running_scores = []
        z = mu  # initialise EWMA statistic to reference mean

        for t in range(n):
            z = lam * x[t] + (1 - lam) * z
            score = abs(z - mu)

            if score > h and t > 0:
                change_points.append(t - 1)
                alarm_scores.append(score)
                mu = self._segment_mean(x, t, warmup)
                z = mu  # re-initialise to new segment mean
                running_scores.append(0.0)
            else:
                running_scores.append(score)

        return change_points, alarm_scores, running_scores

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def _predict(self, X):
        """Detect change points via the EWMA control chart.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of int
            Sorted iloc positions of detected change points.  Each value is
            the last index of the left (pre-change) segment, matching the
            convention used by ``BinarySegmentation`` and ``CUSUM``.
        """
        x = X.to_numpy(dtype=float)
        change_points, _, _ = self._run_ewma(x)
        return pd.Series(change_points, dtype="int64")

    def _predict_scores(self, X):
        r"""Return sparse EWMA scores, one per detected change point.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of float
            One score per detected change point, equal to
            :math:`|Z_t - \mu|` at the step that fired the alarm.
        """
        x = X.to_numpy(dtype=float)
        _, alarm_scores, _ = self._run_ewma(x)
        return pd.Series(alarm_scores, dtype=float)

    def _transform_scores(self, X):
        r"""Return the running EWMA deviation statistic at every timepoint.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of float
            :math:`|Z_t - \mu|` at each index of ``X``.  Resets to 0.0
            immediately after each alarm.  Values above ``h`` indicate a
            detected change point at the *previous* index.
        """
        x = X.to_numpy(dtype=float)
        _, _, running_scores = self._run_ewma(x)
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
        params0 = {"lam": 0.2, "h": 1.0}
        params1 = {"lam": 0.5, "h": 2.0, "target": 0.0, "warmup_len": 5}
        return [params0, params1]
