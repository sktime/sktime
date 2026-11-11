#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    EPS,
    geometric_mean_squared_error,
)


class GeometricMeanSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean squared error (GMSE) or Root geometric mean squared error (RGMSE).

    If ``square_root`` is False then calculates GMSE and if ``square_root`` is True
    then RGMSE is calculated. Both GMSE and RGMSE return non-negative floating
    point. The best value is approximately zero, rather than zero.

    Like MSE and MdSE, GMSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE and RdMSE. Because GMSE and RGMSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than GMAE.

    ``evaluate_by_index`` returns, at each time index, the jackknife
    pseudo-value of the GMSE (or RGMSE, if ``square_root`` is True),
    ``n * theta - (n - 1) * theta_i``, where ``theta`` is the metric over
    all time indices, and ``theta_i`` is the metric with the i-th time
    index removed. The pseudo-values are not themselves metric values
    and, unlike the GMSE, they can be negative.

    Parameters
    ----------
    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root geometric mean squared error (RGMSE)
        If False, returns geometric mean squared error (GMSE)

    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable errors are returned.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call to the metric object averages over time points,
          equivalent to a call of the ``evaluate`` method.
        * If ``True``, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_absolute_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of ``y_true`` and ``y_pred`` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when ``y_true`` equals ``y_pred``)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import GeometricMeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(2.80399089461488e-07)
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.000529527232030127)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5000000000115499)
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5000024031086919)
    >>> gmse = GeometricMeanSquaredError(multioutput='raw_values')
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    array([2.30997255e-11, 1.00000000e+00])
    >>> rgmse = GeometricMeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rgmse(y_true, y_pred)# doctest: +SKIP
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7])
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7000000000069299)
    >>> rgmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7000014418652152)
    """

    func = geometric_mean_squared_error

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        Parameters
        ----------
        y_true : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_pred : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Predicted values to evaluate.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).

            * pd.Series if self.multioutput="uniform_average" or array-like;
              index is equal to index of y_true;
              entry at index i is metric at time i, averaged over variables.
            * pd.DataFrame if self.multioutput="raw_values";
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j.
        """
        multioutput = self.multioutput

        errors = y_true - y_pred
        # convert to float first, as _check_reg_targets does for evaluate,
        # otherwise squaring integer errors can overflow (or wrap negative)
        errors = errors.astype("float64")
        # zero errors are replaced by EPS, as in geometric_mean_squared_error,
        # so that the logarithms below are finite and consistent with evaluate
        errors = errors.mask(errors == 0.0, EPS)
        squared_errors = errors**2

        n = squared_errors.shape[0]
        if n == 1:
            raise ValueError(
                "evaluate_by_index requires at least two time points, "
                "jackknife pseudo-values are not defined for a single time point."
            )

        # leave-one-out geometric mean, in closed form in log-space:
        # the mean of the logs without the i-th point is
        # (sum of logs - log(s_i)) / (n - 1)
        # squared errors can still be 0 if squaring underflowed (zero errors
        # were replaced by EPS above), or inf if squaring overflowed.
        # geometric_mean_squared_error, via scipy's gmean, maps any remaining
        # 0 in a column to a geometric mean of 0, and any remaining inf to
        # inf, so both are counted per column and handled explicitly below,
        # to agree with evaluate on such inputs as well.
        sqe = squared_errors.values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_sqe = np.log(sqe)
        is_zero = sqe == 0.0
        is_inf = np.isinf(sqe)
        is_finite_positive = ~(is_zero | is_inf | np.isnan(sqe))
        log_sqe_finite = np.where(is_finite_positive, log_sqe, 0.0)
        log_sum = log_sqe_finite.sum(axis=0)
        n_zero = is_zero.sum(axis=0)
        n_inf = is_inf.sum(axis=0)

        gmse = np.exp(log_sum / n)
        gmse = np.where(n_zero > 0, 0.0, gmse)
        gmse = np.where((n_zero == 0) & (n_inf > 0), np.inf, gmse)
        gmse = np.where((n_zero > 0) & (n_inf > 0), np.nan, gmse)

        loo_zero = n_zero - is_zero
        loo_inf = n_inf - is_inf
        gmse_jackknife = np.exp((log_sum - log_sqe_finite) / (n - 1))
        gmse_jackknife = np.where(loo_zero > 0, 0.0, gmse_jackknife)
        gmse_jackknife = np.where(
            (loo_zero == 0) & (loo_inf > 0), np.inf, gmse_jackknife
        )
        gmse_jackknife = np.where(
            (loo_zero > 0) & (loo_inf > 0), np.nan, gmse_jackknife
        )

        if self.square_root:
            gmse = gmse**0.5
            gmse_jackknife = gmse_jackknife**0.5

        pseudo_values = n * gmse - (n - 1) * gmse_jackknife
        pseudo_values = pd.DataFrame(
            pseudo_values, index=squared_errors.index, columns=squared_errors.columns
        )

        # NaN input propagates to the entire column, as in
        # GeometricMeanAbsoluteError: evaluate rejects non-finite input via
        # _check_reg_targets, but no _evaluate_by_index in the framework
        # validates finiteness, and finite pseudo-values must not be
        # silently fabricated from the remaining points of the column
        nan_columns = squared_errors.columns[squared_errors.isna().any(axis=0)]
        if len(nan_columns) > 0:
            pseudo_values[nan_columns] = np.nan

        pseudo_values = self._get_weighted_df(pseudo_values, **kwargs)

        return self._handle_multioutput(pseudo_values, multioutput)

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput, multilevel=multilevel, by_index=by_index
        )

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
        params2 = {"square_root": True}
        return [params1, params2]
