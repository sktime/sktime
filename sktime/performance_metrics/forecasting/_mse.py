#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class MeanSquaredError(BaseForecastingErrorMetric):
    r"""Mean squared error (MSE) or root mean squared error (RMSE).

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n` (in :math:`mathbb{R}`),
    at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns:

    * if ``square_root`` is False, the Mean Squared Error,
      :math:`\frac{1}{n}\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2`
    * if ``square_root`` is True, the Root Mean Squared Error,
      :math:`\sqrt{\frac{1}{n}\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}`

    MSE and RMSE are both non-negative floating point, lower values are better.
    The lowest possible value is 0.0.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below. If ``square_root`` is True, averages
    are taken over square roots of squared errors.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`:

    * if ``square_root`` is False, the squared error at that time index,
      :math:`\left(y_i - \widehat{y}_i\right)^2`,
      for all time indices :math:`t_1, \dots, t_n` in the input.
    * if ``square_root`` is True, the jackknife pseudo-value of the RMSE
      at that time index, :math:`n * \bar{\varepsilon} - (n-1) * \varepsilon_i`,
      where :math:`\bar{\varepsilon}` is the RMSE over all time indices,
      and :math:`\varepsilon_i` is the RMSE with the i-th time index removed,
      i.e., using values :math:`y_1, \dots, y_{i-1}, y_{i+1}, \dots, y_n`,
      and :math:`\widehat{y}_1, \dots, \widehat{y}_{i-1}, \widehat{y}_{i+1}, \dots, \widehat{y}_n`.

    MSE is measured in squared units of the input data, and RMSE is on the
    same scale as the data. Because MSE and RMSE square the forecast error
    rather than taking the absolute value, they penalize large errors more than
    MAE.

    Parameters
    ----------
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
    MeanAbsoluteError
    MedianAbsoluteError
    MedianSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mse = MeanSquaredError()
    >>> mse(y_true, y_pred)
    np.float64(0.4125)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mse(y_true, y_pred)
    np.float64(0.7083333333333334)
    >>> rmse = MeanSquaredError(square_root=True)
    >>> rmse(y_true, y_pred)
    np.float64(0.8227486121839513)
    >>> rmse = MeanSquaredError(multioutput='raw_values')
    >>> rmse(y_true, y_pred)
    array([0.41666667, 1.        ])
    >>> rmse = MeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rmse(y_true, y_pred)
    array([0.64549722, 1.        ])
    >>> rmse = MeanSquaredError(multioutput=[0.3, 0.7])
    >>> rmse(y_true, y_pred)
    np.float64(0.825)
    >>> rmse = MeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmse(y_true, y_pred)
    np.float64(0.8936491673103708)
    """  # noqa: E501

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
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

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                value is metric averaged over variables (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                i-th entry is metric calculated for i-th variable
        """
        multioutput = self.multioutput

        raw_values = (y_true - y_pred) ** 2
        raw_values = self._get_weighted_df(raw_values, **kwargs)
        msqe = raw_values.mean()

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

        raw_values = (y_true - y_pred) ** 2

        if self.square_root:
            n = raw_values.shape[0]
            mse = raw_values.mean(axis=0)
            rmse = mse.pow(0.5)
            sqe_sum = raw_values.sum(axis=0)
            mse_jackknife = (sqe_sum - raw_values) / (n - 1)
            rmse_jackknife = mse_jackknife.pow(0.5)
            pseudo_values = n * rmse - (n - 1) * rmse_jackknife
        else:
            pseudo_values = raw_values

        pseudo_values = self._get_weighted_df(pseudo_values, **kwargs)

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return pseudo_values

            if multioutput == "uniform_average":
                return pseudo_values.mean(axis=1)

        # else, we expect multioutput to be array-like
        return pseudo_values.dot(multioutput)

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
