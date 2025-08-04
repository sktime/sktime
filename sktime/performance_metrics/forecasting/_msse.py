#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import pandas as pd

from sktime.performance_metrics.forecasting._base import (
    BaseForecastingErrorMetricFunc,
    _ScaledMetricTags,
)
from sktime.performance_metrics.forecasting._functions import mean_squared_scaled_error


class MeanSquaredScaledError(_ScaledMetricTags, BaseForecastingErrorMetricFunc):
    """Mean squared scaled error (MSSE) or root mean squared scaled error (RMSSE).

    If ``square_root`` is False then calculates MSSE, otherwise calculates RMSSE if
    ``square_root`` is True. Both MSSE and RMSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared variant of the MASE loss metric.  Like MASE and other
    scaled performance metrics this scale-free metric can be used to compare
    forecast methods on a single series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of data.
    square_root : bool, default = False
        Whether to take the square root of the metric

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.

        * If array-like, values used as weights to average the errors.
        * If ``'raw_values'``,
          returns a full set of errors in case of multioutput input.
        * If ``'uniform_average'``,
          errors of all outputs are averaged with uniform weight.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Determines averaging over time points in direct call to metric object.

        * If False, direct call to the metric object averages over time points,
          equivalent to a call of the``evaluate`` method.
        * If True, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

    See Also
    --------
    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MedianSquaredScaledError

    References
    ----------
    M5 Competition Guidelines.

    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanSquaredScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> rmsse = MeanSquaredScaledError(square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    np.float64(0.20568833780186058)
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> rmsse(y_true, y_pred, y_train=y_train)
    np.float64(0.15679361328058636)
    >>> rmsse = MeanSquaredScaledError(multioutput='raw_values', square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    array([0.11215443, 0.20203051])
    >>> rmsse = MeanSquaredScaledError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    np.float64(0.17451891814894502)
    """

    func = mean_squared_scaled_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
        square_root=False,
        by_index=False,
    ):
        self.sp = sp
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        By default this uses evaluate_by_index, taking arithmetic mean over time points.

        Parameters
        ----------
        y_true : time series in sktime compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed
        y_train : pandas container, passed via `kwargs`
            Historical (in-sample) values used to compute the scaling denominator.
            Must have at least `sp + 1` rows.

        Returns
        -------
        msse : float or pandas.Series
            If `multioutput="uniform_average"` (the default), returns a scalar float.
            If `multioutput="raw_values"`, returns a Series indexed by column names,
            giving one MSSE/RMSSE per series.
        """
        multioutput = self.multioutput

        raw_values = (y_true - y_pred) ** 2

        y_train = kwargs["y_train"]
        differences = []
        N = len(y_train)
        s = self.sp
        # compute (y[t] - y[t-s])^2 and collect these values
        for i in range(0, N - s):
            # y_train.iloc[i + s]: value at time t
            # y_train.iloc[i]: value at time t - s
            differences.append((y_train.iloc[i + s] - y_train.iloc[i]) ** 2)
        # Stack the list of Series into a DataFrame, then take the column-wise mean
        # This gives the in-sample naive MSE (denominator) for each series
        seasonal_mse = pd.DataFrame(differences).mean(axis=0)
        # Divide the raw squared forecast errors (numerator) by the
        # seasonal MSE (denominator)
        scaled = raw_values.divide(seasonal_mse, axis=1)
        # apply weights
        scaled = self._get_weighted_df(scaled, **kwargs)
        msse = scaled.mean()

        #  optional sqrt
        if self.square_root:
            msse = msse.pow(0.5)

        #  aggregate across outputs
        return self._handle_multioutput(msse, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        Parameters
        ----------
        y_true : time series in sktime compatible pandas based data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series scitype: pd.DataFrame
            Panel scitype: pd.DataFrame with 2-level row MultiIndex
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed
        y_train : pd.DataFrame, passed via `kwargs`
            Historical in-sample values used to compute the scaling denominator.
            Must have at least `sp + 1` rows.

        Returns
        -------
        msse : pd.Series or pd.DataFrame
            Per-time-point MSSE or RMSSE pseudo-values, before time-averaging.
            - If `multioutput="uniform_average"` (default) or array-like weights:
                returns a pd.Series of length `h` (forecast horizon), indexed as
                `y_true`.
                Entry at index `i` is the MSSE (or RMSSE pseudo-value) at time `i`,
                averaged across variables.
            - If `multioutput="raw_values"`:
                returns a pd.DataFrame of shape `(h, d)`, with the same index and
                columns as `y_true`. The entry at `(i, j)` is the MSSE (or RMSSE
                pseudo-value) for series `j` at time `i`.
        """
        multioutput = self.multioutput

        raw_values = (y_true - y_pred) ** 2

        y_train = kwargs["y_train"]
        differences = []
        N = len(y_train)
        s = self.sp
        for i in range(0, N - s):
            differences.append((y_train.iloc[i + s] - y_train.iloc[i]) ** 2)

        seasonal_mse = pd.DataFrame(differences).mean(axis=0)
        scaled = raw_values.divide(seasonal_mse, axis=1)

        if self.square_root:
            msse_full = scaled.mean(axis=0)
            rmsse_full = msse_full.pow(0.5)

            # Determine the number of time points
            n = scaled.shape[0]
            # Sum the scaled errors along time for each series
            sum_scaled = scaled.sum(axis=0)
            # Calculate the leave-one-out (LOO) MSSE
            msse_loo = (sum_scaled - scaled) / (n - 1)
            # Convert the leave-one-out MSSE to RMSSE by taking the square root
            rmsse_loo = msse_loo.pow(0.5)

            # jackknife pseudo-values: n*full - (n-1)*loo
            pseudo = pd.DataFrame(
                n * rmsse_full - (n - 1) * rmsse_loo,
                index=scaled.index,
                columns=scaled.columns,
            )
        else:
            # for MSSE, raw scaled errors are the pseudo-values
            pseudo = scaled

        #  apply weights
        pseudo = self._get_weighted_df(pseudo, **kwargs)

        # aggregate across series per time point
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return pseudo
            # uniform average over series
            return pseudo.mean(axis=1)
        # array-like weights
        return pseudo.dot(multioutput)

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
        params1 = {}
        params2 = {"sp": 2, "square_root": True}
        return [params1, params2]
