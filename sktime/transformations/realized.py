"""Realized measures of high-frequency returns: realized variance, bipower variation.

Each transformer cuts a series of intraday returns into periods (a fixed number of
bars, or calendar periods of a time index) and returns one value per period.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Mattbusel"]
__all__ = ["RealizedVariance", "BipowerVariation", "JumpVariation"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

# E|Z| for Z ~ N(0, 1); bipower variation is scaled by MU1 ** -2 = pi / 2
MU1 = np.sqrt(2.0 / np.pi)


def _rv_blocks(blocks):
    """Realized variance along axis 1 of a (n_periods, n_bars, n_cols) array."""
    return np.sum(blocks**2, axis=1)


def _bv_blocks(blocks):
    """Bipower variation along axis 1 of a (n_periods, n_bars, n_cols) array."""
    a = np.abs(blocks)
    return np.sum(a[:, 1:] * a[:, :-1], axis=1) / MU1**2


class _RealizedMeasure(BaseTransformer):
    """Base class: cut intraday returns into periods and reduce each period."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["Mattbusel"],
        "maintainers": ["Mattbusel"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:unequal_length:removes": True,
        "capability:missing_values": False,
        "capability:categorical_in_X": False,
        "transform-returns-same-time-index": False,
    }

    def __init__(self, freq="D", annualize=False, trading_days=252):
        self.freq = freq
        self.annualize = annualize
        self.trading_days = trading_days
        super().__init__()

    def _reduce(self, blocks):
        """Reduce a (n_periods, n_bars, n_cols) array to (n_periods, n_cols)."""
        raise NotImplementedError("abstract method")

    def _annualize(self, values):
        """Scale per-period values by ``trading_days`` if ``annualize`` is True."""
        if self.annualize:
            return values * self.trading_days
        return values

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            intraday returns, one column per variable
        y : ignored, present for interface compatibility

        Returns
        -------
        Xt : pd.DataFrame
            one row per complete period, same columns as ``X``.
            With an integer ``freq``, rows are labelled by the last bar of the
            period. With a string ``freq``, rows are labelled by the period:
            the period start time stamp for a ``DatetimeIndex``,
            or the ``Period`` for a ``PeriodIndex``.
        """
        values = X.to_numpy(dtype=float)
        n_cols = X.shape[1]

        if isinstance(self.freq, str):
            index, groups = self._calendar_groups(X.index)
            rows = [self._reduce(values[g][None, :, :])[0] for g in groups]
            out = np.vstack(rows) if rows else np.empty((0, n_cols))
        else:
            freq = int(self.freq)
            if freq < 1:
                raise ValueError(
                    f"freq must be a positive integer or a pandas offset alias, "
                    f"but found {self.freq}"
                )
            n_periods = len(X) // freq
            blocks = values[: n_periods * freq].reshape(n_periods, freq, n_cols)
            out = self._reduce(blocks)
            index = X.index[freq - 1 : n_periods * freq : freq]

        out = self._annualize(out)
        return pd.DataFrame(out, index=index, columns=X.columns)

    def _calendar_groups(self, index):
        """Group positions of ``index`` by calendar period ``self.freq``."""
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is not None:
                index = index.tz_localize(None)
            periods = index.to_period(self.freq)
        elif isinstance(index, pd.PeriodIndex):
            periods = index.asfreq(self.freq)
        else:
            raise TypeError(
                f"freq={self.freq!r} groups by calendar period and requires a "
                "DatetimeIndex or PeriodIndex, but found "
                f"{type(index).__name__}. Pass an integer freq (bars per period) "
                "for other index types."
            )
        codes, uniques = pd.factorize(periods, sort=True)
        groups = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        uniques = pd.PeriodIndex(uniques)
        if isinstance(index, pd.DatetimeIndex):
            uniques = uniques.to_timestamp()
        return uniques, groups

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"freq": 1}
        params2 = {"freq": 5}
        params3 = {"freq": 3, "annualize": True, "trading_days": 365}
        return [params1, params2, params3]


class RealizedVariance(_RealizedMeasure):
    r"""Realized variance per period of intraday returns.

    For intraday returns :math:`r_1, \dots, r_n` of one period (usually a trading
    day), the realized variance is :math:`RV = \sum_{i=1}^n r_i^2`.
    It is a consistent estimator of the quadratic variation of the log price
    over the period, and the usual target of realized volatility forecasting,
    see [1]_.

    ``transform`` cuts ``X`` into periods and returns one value per period
    and variable.

    Parameters
    ----------
    freq : int or str, default="D"
        How the series of intraday returns is cut into periods.

        * int: number of bars per period, e.g., 78 five-minute bars in a
          6.5 hour US equity session. A trailing incomplete period is dropped.
          Rows of the output are labelled by the last bar of each period.
        * str: a pandas offset alias such as ``"D"``, ``"W"`` or ``"M"``, used to
          group a ``DatetimeIndex`` or ``PeriodIndex`` by calendar period.
          Rows are labelled by the period start (``DatetimeIndex``) or by the
          period (``PeriodIndex``).

    annualize : bool, default=False
        whether to multiply each value by ``trading_days``
    trading_days : int, default=252
        periods per year, used if ``annualize`` is True

    See Also
    --------
    BipowerVariation : jump-robust estimate of integrated variance
    JumpVariation : part of realized variance due to jumps

    References
    ----------
    .. [1] Andersen, T. G. and Bollerslev, T. (1998). Answering the skeptics:
       yes, standard volatility models do provide accurate forecasts.
       International Economic Review, 39(4), 885-905.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.realized import RealizedVariance
    >>> r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01])
    >>> RealizedVariance(freq=3).fit_transform(r)
    2    0.0014
    5    0.0002
    dtype: float64
    """

    def _reduce(self, blocks):
        return _rv_blocks(blocks)


class BipowerVariation(_RealizedMeasure):
    r"""Bipower variation per period of intraday returns.

    For intraday returns :math:`r_1, \dots, r_n` of one period, the bipower
    variation is

    .. math:: BV = \mu_1^{-2} \sum_{i=2}^n |r_i| |r_{i-1}|,

    where :math:`\mu_1 = E|Z| = \sqrt{2 / \pi}` for standard normal :math:`Z`.
    Under a continuous semimartingale, ``BV`` converges to the integrated variance,
    like realized variance. Unlike realized variance, it is not moved by jumps of
    finite activity, see [1]_. No small-sample correction is applied.

    ``transform`` cuts ``X`` into periods and returns one value per period
    and variable.

    Parameters
    ----------
    freq : int or str, default="D"
        How the series of intraday returns is cut into periods.

        * int: number of bars per period, e.g., 78 five-minute bars in a
          6.5 hour US equity session. A trailing incomplete period is dropped.
          Rows of the output are labelled by the last bar of each period.
        * str: a pandas offset alias such as ``"D"``, ``"W"`` or ``"M"``, used to
          group a ``DatetimeIndex`` or ``PeriodIndex`` by calendar period.
          Rows are labelled by the period start (``DatetimeIndex``) or by the
          period (``PeriodIndex``).

    annualize : bool, default=False
        whether to multiply each value by ``trading_days``
    trading_days : int, default=252
        periods per year, used if ``annualize`` is True

    See Also
    --------
    RealizedVariance : realized variance, includes jumps
    JumpVariation : part of realized variance due to jumps

    References
    ----------
    .. [1] Barndorff-Nielsen, O. E. and Shephard, N. (2004). Power and bipower
       variation with stochastic volatility and jumps. Journal of Financial
       Econometrics, 2(1), 1-37.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.realized import BipowerVariation
    >>> r = pd.Series([0.01, -0.02, 0.03, 0.0])
    >>> BipowerVariation(freq=4).fit_transform(r).round(6)
    3    0.001257
    dtype: float64
    """

    def _reduce(self, blocks):
        return _bv_blocks(blocks)


class JumpVariation(_RealizedMeasure):
    r"""Jump variation per period of intraday returns.

    The jump variation is the part of realized variance not explained by bipower
    variation, truncated at zero: :math:`JV = \max(RV - BV, 0)`,
    see [1]_. With ``relative=True``, the relative jump measure
    :math:`\max(RV - BV, 0) / RV` is returned instead, which lies in :math:`[0, 1]`.

    ``transform`` cuts ``X`` into periods and returns one value per period
    and variable.

    Parameters
    ----------
    freq : int or str, default="D"
        How the series of intraday returns is cut into periods.

        * int: number of bars per period, e.g., 78 five-minute bars in a
          6.5 hour US equity session. A trailing incomplete period is dropped.
          Rows of the output are labelled by the last bar of each period.
        * str: a pandas offset alias such as ``"D"``, ``"W"`` or ``"M"``, used to
          group a ``DatetimeIndex`` or ``PeriodIndex`` by calendar period.
          Rows are labelled by the period start (``DatetimeIndex``) or by the
          period (``PeriodIndex``).

    annualize : bool, default=False
        whether to multiply each value by ``trading_days``.
        Has no effect if ``relative=True``.
    trading_days : int, default=252
        periods per year, used if ``annualize`` is True
    relative : bool, default=False
        whether to return the relative jump measure ``max(RV - BV, 0) / RV``.
        Periods with ``RV = 0`` get the value 0.

    See Also
    --------
    RealizedVariance : realized variance, includes jumps
    BipowerVariation : jump-robust estimate of integrated variance

    References
    ----------
    .. [1] Barndorff-Nielsen, O. E. and Shephard, N. (2006). Econometrics of
       testing for jumps in financial economics using bipower variation.
       Journal of Financial Econometrics, 4(1), 1-30.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.realized import JumpVariation
    >>> r = pd.Series([0.001, -0.001, 0.05, 0.001, -0.001, 0.001])
    >>> JumpVariation(freq=6, relative=True).fit_transform(r).round(3)
    5    0.935
    dtype: float64
    """

    def __init__(self, freq="D", annualize=False, trading_days=252, relative=False):
        self.relative = relative
        super().__init__(freq=freq, annualize=annualize, trading_days=trading_days)

    def _reduce(self, blocks):
        rv = _rv_blocks(blocks)
        jv = np.maximum(rv - _bv_blocks(blocks), 0.0)
        if self.relative:
            safe_rv = np.where(rv > 0, rv, 1.0)
            return np.where(rv > 0, jv / safe_rv, 0.0)
        return jv

    def _annualize(self, values):
        if self.relative:
            return values
        return super()._annualize(values)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"freq": 2}
        params2 = {"freq": 4, "relative": True}
        params3 = {"freq": 3, "annualize": True}
        return [params1, params2, params3]
