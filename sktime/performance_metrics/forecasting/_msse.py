#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import (
    BaseForecastingErrorMetricFunc,
    _ScaledMetricTags,
)
from sktime.performance_metrics.forecasting._common import _fraction, _pseudovalues_sqrt
from sktime.performance_metrics.forecasting._functions import mean_squared_scaled_error


class MeanSquaredScaledError(_ScaledMetricTags, BaseForecastingErrorMetricFunc):
    r"""Mean squared scaled error (MSSE) or root mean squared scaled error (RMSSE).
  
    If ``square_root`` is False then calculates MSSE, otherwise calculates RMSSE if
    ``square_root`` is True. Both MSSE and RMSSE output is non-negative floating
    point.
  
    Both MSSE and RMSSE are unitless metrics.
    Lower values are better, and the lowest possible value is 0.0.

    For a univariate, non-hierarchical sample of true values
    :math:`y_1, \dots, y_n`, predicted values
    :math:`\widehat{y}_1, \dots, \widehat{y}_n` (in :math:`\mathbb{R}`),
    and training values :math:`y_{\text{train}, 1}, \dots, y_{\text{train}, m}`,
    at time indices :math:`t_1, \dots, t_n`, ``evaluate`` or call returns:

    * if ``square_root`` is False, the Mean Squared Scaled Error (MSSE),
      :math:`\frac{\frac{1}{n}\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}
      {\frac{1}{m-s}\sum_{j=1}^{m-s} \left(y_{\text{train}, j+s} - y_{\text{train}, j}\right)^2}`
    * if ``square_root`` is True, the Root Mean Squared Scaled Error (RMSSE),
      :math:`\sqrt{\frac{\frac{1}{n}\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}
      {\frac{1}{m-s}\sum_{j=1}^{m-s} \left(y_{\text{train}, j+s} - y_{\text{train}, j}\right)^2}}`

    MSSE and RMSSE are both non-negative floating point. Lower values are better,
    and the lowest possible value is 0.0.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below. If ``square_root`` is True, averages are taken
    over square roots of scaled squared errors.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`:

    * if ``square_root`` is False, the scaled squared error at that time index,
      :math:`\frac{\left(y_i - \widehat{y}_i\right)^2}
      {\frac{1}{m-s}\sum_{j=1}^{m-s} \left(y_{\text{train}, j+s} - y_{\text{train}, j}\right)^2}`,
      for all time indices :math:`t_1, \dots, t_n` in the input.
    * if ``square_root`` is True, the jackknife pseudo-value of the RMSSE
      at that time index, :math:`n * \bar{\varepsilon} - (n-1) * \varepsilon_i`,
      where :math:`\bar{\varepsilon}` is the RMSSE over all time indices,
      and :math:`\varepsilon_i` is the RMSSE with the i-th time index removed,
      i.e., using values :math:`y_1, \dots, y_{i-1}, y_{i+1}, \dots, y_n`,
      and :math:`\widehat{y}_1, \dots, \widehat{y}_{i-1}, \widehat{y}_{i+1}, \dots, \widehat{y}_n`.

    MSSE and RMSSE are scale-free metrics, making them suitable for comparing
    forecast methods across series. They are also robust to intermittent-demand
    series, as they avoid infinite or undefined values unless the training data
    is flat. In such cases, the function returns a large value instead of infinity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of data.
    square_root : bool, default = False
        Whether to take the square root of the metric
    eps: float, default = None
        Numerical epsilon used to avoid division by zero or instability.
        Absolute values of the denominator smaller than ``eps`` are replaced
        by ``eps``. If None, defaults to ``np.finfo(np.float64).eps``.

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
    """  # noqa: E501

    func = mean_squared_scaled_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
        eps=None,
        square_root=False,
        by_index=False,
    ):
        self.sp = sp
        self.eps = eps
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

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
        loss : float or np.ndarray
            Calculated metric, possibly averaged by variable given ``multioutput``.

            * float if ``multioutput="uniform_average" or array-like,
              Value is metric averaged over variables and levels (see class docstring)
            * ``np.ndarray`` of shape ``(y_true.columns,)``
              if `multioutput="raw_values"``
              i-th entry is the, metric calculated for i-th variable
        """
        multioutput = self.multioutput

        raw_values = (y_true - y_pred) ** 2

        y_train = kwargs["y_train"]
        sp = self.sp
        denominator = y_train.diff(sp).pow(2).mean(axis=0)
        seasonal_mse = _fraction(
            enumerator=raw_values,
            denominator=denominator,
            eps=self.eps,
        )

        scaled = self._get_weighted_df(seasonal_mse, **kwargs)
        msse = scaled.mean()

        if self.square_root:
            msse = msse.pow(0.5)

        return self._handle_multioutput(msse, multioutput)

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

        raw_values = (y_true - y_pred) ** 2

        y_train = kwargs["y_train"]
        sp = self.sp
        denominator = y_train.diff(sp).pow(2).mean(axis=0)

        scaled = _fraction(
            enumerator=raw_values,
            denominator=denominator,
            eps=self.eps,
        )

        if self.square_root:
            pseudo = _pseudovalues_sqrt(scaled)
        else:
            pseudo = scaled

        pseudo = self._get_weighted_df(pseudo, **kwargs)

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return pseudo
            return pseudo.mean(axis=1)

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
