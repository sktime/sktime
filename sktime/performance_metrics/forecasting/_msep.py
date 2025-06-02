#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc


class MeanSquaredErrorPercentage(BaseForecastingErrorMetricFunc):
    r"""Mean Squared Error Percentage (MSE%) and root-MSE% forecasting error metrics.

    Calculates the mean squared error percentage between the true and predicted values.
    Optionally, the root mean squared error percentage (RMSE%) can be computed by
    setting ``square_root=True``.

    The Mean Squared Error Percentage (MSE%) is calculated as:

    .. math::
        \\text{MSE%} = \\frac{ \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 }
                            { \left| \\sum_{i=1}^{n} \\frac{y_i}{n} \right| }

    where:
    - \\( y_i \\) are the true values,
    - \\( \\hat{y}_i \\) are the predicted values,
    - \\( n \\) is the number of observations.

    If ``square_root`` is set to True,
    the Root Mean Squared Error Percentage (RMSE%) is computed:

    .. math::
        \\text{RMSE%} = \\sqrt{ \\text{MSE%} }

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.
    by_index : bool, default=False
        Determines averaging over time points in direct call to metric object.

        * If False, direct call to the metric object averages over time points,
          equivalent to a call of the``evaluate`` method.
        * If True, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.
    """

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
        self.multioutput = multioutput

        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate(self, y_true, y_pred, **kwargs):
        r"""
        Evaluate the Mean Squared Error Percentage (MSE%) between `y_true` and `y_pred`.

        Parameters
        ----------
        y_true : pd.Series or pd.DataFrame
            Ground truth (actual) target values.
            Can be a Series or DataFrame for univariate or multivariate forecasts.

        y_pred : pd.Series or pd.DataFrame
            Forecasted target values.
            Must have the same shape as `y_true`.

        Returns
        -------
        loss : float or pd.Series
            The calculated Mean Squared Error Percentage.
            - If `multioutput='raw_values'`, returns a Series with the MSPE for
            each output.
            - Otherwise, returns a scalar value representing the aggregated MSPE.
        """
        multioutput = self.multioutput
        raw_values = (y_true - y_pred) ** 2
        raw_values = self._get_weighted_df(raw_values, **kwargs)
        num = raw_values.mean()
        denom = y_true.mean().abs()

        msqe = num / denom

        if self.square_root:
            msqe = msqe.pow(0.5)

        return self._handle_multioutput(msqe, multioutput)

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

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput

        raw_values_mse = (y_true - y_pred) ** 2
        raw_values_p = y_true

        # what we need to do is efficiently
        # compute msqe but using data with the i-th time point removed
        # msqe[i] = msqe(all data minus i-th time point)

        n = raw_values_mse.shape[0]

        num_mean = raw_values_mse.mean()
        denom_mean = raw_values_p.mean()

        num_jk = num_mean * (1 + 1 / (n - 1)) - raw_values_mse / (n - 1)
        denom_jk = denom_mean * (1 + 1 / (n - 1)) - raw_values_p / (n - 1)

        msep_jk = num_jk / denom_jk
        msep = num_mean / denom_mean

        if self.square_root:
            msep_jk = msep_jk.pow(0.5)
            msep = msep.pow(0.5)

        pseudo_values = n * msep - (n - 1) * msep_jk
        pseudo_values = self._get_weighted_df(pseudo_values, **kwargs)

        return self._handle_multioutput(pseudo_values, multioutput)

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
