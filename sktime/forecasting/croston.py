# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Croston's Forecasting Method."""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from sktime.forecasting.base import BaseForecaster


def _exp_smooth(x, start, smoothing):
    """Exponentially smooth ``x``, starting from ``start``.

    Computes ``out[0] = smoothing * x[0] + (1 - smoothing) * start`` and
    ``out[k] = smoothing * x[k] + (1 - smoothing) * out[k - 1]`` for ``k >= 1``,
    in compiled code. This is exactly the recursion Croston's method applies
    at non-zero demand observations.
    """
    zi = np.asarray([start * (1.0 - smoothing)])
    return lfilter([smoothing], [1.0, -(1.0 - smoothing)], x, zi=zi)[0]


class Croston(BaseForecaster):
    r"""Croston's method for forecasting intermittent time series.

    Implements the method proposed by Croston in [1]_ and described in [2]_.

    Croston's method is a modification of (vanilla) exponential smoothing to handle
    intermittent time series. A time series is considered intermittent if many
    of its values are zero and the gaps between non-zero entries are not periodic.

    Croston's method will predict a constant value for all future times, so
    Croston's method essentially provides another notion for the average value
    of a time series.

    The method is (equivalent to) the following:

    - Let :math:`v_0,\ldots,v_n` be the non-zero values of the time series
    - Let :math:`v` be the exponentially smoothed average of :math:`v_0,\ldots,v_n`
    - Let :math:`z_0,\ldots,z_n` be the number of consecutive zeros plus 1 between
      the :math:`v_i` in the original time series.
    - Let :math:`z` be the exponentially smoothed average of :math:`z_0,\ldots,z_n`
    - Then the forecast is :math:`\frac{v}{z}`

    The intuition is that :math:`v` is a weighted average of the non-zero time
    series values and :math:`\frac{1}{z}` estimates the probability of getting a
    non-zero value.

    Example to illustrate the :math:`v` and :math:`z` notation.

    - If the original time series is :math:`0,0,2,7,0,0,0,-5` then:

        - The :math:`v`'s are :math:`2,7,-5`
        - The :math:`z`'s are :math:`3,1,4`

    Parameters
    ----------
    smoothing : float, default = 0.1
        Smoothing parameter in exponential smoothing

    Examples
    --------
    >>> from sktime.forecasting.croston import Croston
    >>> from sktime.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()
    >>> forecaster = Croston(smoothing=0.1)
    >>> forecaster.fit(y)
    Croston(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    See Also
    --------
    ExponentialSmoothing

    References
    ----------
    .. [1] J. D. Croston. Forecasting and stock control for intermittent demands.
       Operational Research Quarterly (1970-1977), 23(3):pp. 289-303, 1972.
    .. [2] N. Vandeput. Forecasting Intermittent Demand with the Croston Model.

       https://towardsdatascience.com/croston-forecast-model-for-intermittent-demand-360287a17f5f
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "Riyabelle25",
        "maintainers": "Riyabelle25",
        # estimator type
        # --------------
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "capability:exogenous": False,
        "capability:update": True,  # can estimator update its parameters with new data?
        "y_inner_mtype": "pd.DataFrame",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, smoothing=0.1):
        # hyperparameter
        self.smoothing = smoothing
        self._f = None
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        n_timepoints = len(y)  # Historical period: i.e the input array's length
        smoothing = self.smoothing

        y = y.to_numpy().flatten()  # Transform the input into a numpy array

        # Fit the parameters: level (q), periodicity (a) and forecast (f).
        #
        # The recursion below only changes state at non-zero observations, so
        # it is equivalent to exponential smoothing over the subsequence of
        # non-zero demands, with the smoothed values held constant in between.
        # The vectorized form computes the same values without the
        # Python-level loop over all time points, which dominates fit time
        # on long series.
        demand_idx = np.flatnonzero(y > 0)

        if len(demand_idx) == 0:
            # no non-zero demand: same degenerate state the loop would produce,
            # as argmax on a series without positive values returns 0
            self._f = np.full(n_timepoints + 1, y[0])
            self._q_last = y[0]
            self._a_last = 1.0
            self._p = 1 + n_timepoints
            self._seen_demand = False
            return self

        demands = y[demand_idx]
        # periods between consecutive non-zero demands; the first entry also
        # counts the leading zeros plus one, mirroring the loop's p
        gaps = np.diff(demand_idx, prepend=-1).astype(float)

        q_smooth = _exp_smooth(demands, demands[0], smoothing)
        a_smooth = _exp_smooth(gaps, gaps[0], smoothing)
        ratio = q_smooth / a_smooth

        # f[t] is the smoothed forecast after t observations; it only changes
        # right after a non-zero demand and stays constant otherwise
        f = np.empty(n_timepoints + 1)
        f[0] = demands[0] / gaps[0]
        n_demands_seen = np.searchsorted(
            demand_idx + 1, np.arange(1, n_timepoints + 1), side="right"
        )
        # before the first demand the forecast stays at its initial value
        f[1:] = np.where(n_demands_seen == 0, f[0], ratio[n_demands_seen - 1])
        self._f = f

        # terminal state of the recursion, so that ``_update`` can continue it
        # incrementally instead of refitting from scratch. ``p`` in particular
        # is not recoverable from ``f`` alone, but is needed to smooth the
        # interval estimate at the next non-zero observation.
        self._q_last = q_smooth[-1]
        self._a_last = a_smooth[-1]
        self._p = int(n_timepoints - demand_idx[-1])
        self._seen_demand = True

        return self

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters on new data.

        Continues Croston's recursion from the state persisted by ``_fit``,
        rather than refitting on the full history.

        Parameters
        ----------
        y : pd.DataFrame
            Time series with which to update the forecaster. Contains only the
            new observations, not the full history.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if not update_params:
            return self

        y = y.to_numpy().flatten()
        n_new = len(y)
        if n_new == 0:
            return self

        # If ``_fit`` saw no non-zero demand, its initialization was degenerate
        # (``argmax`` on an all-zero series returns 0), and the recursion state
        # does not correspond to what a fit on the concatenated series would
        # produce. Refitting on the accumulated history restores exactness.
        if not self._seen_demand and np.any(y > 0):
            return self._fit(y=self._y, X=X, fh=None)

        smoothing = self.smoothing
        demand_idx = np.flatnonzero(y > 0)

        if len(demand_idx) == 0:
            # no new demand: the smoothed state is unchanged, only the
            # periods-since-last-demand counter advances
            self._f = np.concatenate([self._f, np.full(n_new, self._f[-1])])
            self._p = self._p + n_new
            return self

        demands = y[demand_idx]
        # gaps[0] continues the counter carried over from the previous data
        gaps = np.diff(demand_idx, prepend=-self._p).astype(float)

        q_smooth = _exp_smooth(demands, self._q_last, smoothing)
        a_smooth = _exp_smooth(gaps, self._a_last, smoothing)
        ratio = q_smooth / a_smooth

        f_new = np.empty(n_new + 1)
        f_new[0] = self._f[-1]
        n_demands_seen = np.searchsorted(
            demand_idx + 1, np.arange(1, n_new + 1), side="right"
        )
        # before the first new demand, the forecast stays at its old value
        f_new[1:] = np.where(n_demands_seen == 0, f_new[0], ratio[n_demands_seen - 1])

        self._f = np.concatenate([self._f, f_new[1:]])
        self._q_last = q_smooth[-1]
        self._a_last = a_smooth[-1]
        self._p = int(n_new - demand_idx[-1])
        self._seen_demand = True

        return self

    def _predict(
        self,
        fh=None,
        X=None,
    ):
        """Predict forecast.

        Parameters
        ----------
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        forecast : pd.series
            Predicted forecasts.
        """
        len_fh = len(self.fh)
        f = self._f

        # Predicting future forecasts:to_numpy()
        y_pred = np.full(len_fh, f[-1])

        index = self.fh.to_absolute_index(self.cutoff)
        return pd.DataFrame(y_pred, index=index, columns=self._get_varnames())

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
        """
        params = [
            {},
            {"smoothing": 0},
            {"smoothing": 0.42},
            {"smoothing": 2},
        ]

        return params
