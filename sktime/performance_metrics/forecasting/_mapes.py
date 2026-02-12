#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Rolling MAD-stabilized symmetric MAPE (msMAPE) metric."""

__author__ = ["michaelellis003"]

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class MeanAbsolutePercentageErrorStabilized(BaseForecastingErrorMetric):
    r"""Rolling MAD-stabilized symmetric mean absolute percentage error (msMAPE).

    A variant of symmetric MAPE that adds a rolling mean absolute deviation (MAD)
    term to the denominator, stabilizing the metric when actual and predicted
    values are near zero.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns

    .. math::
        \text{msMAPE} = \frac{1}{n} \sum_{i=1}^{n}
        \frac{|y_i - \widehat{y}_i|}{(|y_i| + |\widehat{y}_i|) / 2 + S_i}

    where

    .. math::
        S_1 = 0, \qquad
        S_i = \frac{1}{i-1} \sum_{k=1}^{i-1} |y_k - \bar{y}_{i-1}|
        \quad (i \ge 2)

    and :math:`\bar{y}_{i-1} = \frac{1}{i-1} \sum_{k=1}^{i-1} y_k` is the
    rolling mean of the first :math:`i-1` true values.

    The :math:`S_i` term provides stability when both :math:`y_i` and
    :math:`\widehat{y}_i` are near zero, addressing a known weakness of sMAPE.

    To avoid division by zero, any denominator is replaced by ``eps``
    if it is smaller than ``eps``; the value of ``eps`` defaults to
    ``np.finfo(np.float64).eps`` if not specified.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    the stabilized absolute percentage error at that time index,
    :math:`\frac{|y_i - \widehat{y}_i|}{(|y_i| + |\widehat{y}_i|) / 2 + S_i}`,
    for all time indices :math:`t_1, \dots, t_n` in the input.

    Parameters
    ----------
    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable errors are
          returned.

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
        * If ``True``, direct call to the metric object evaluates the metric at
          each time point, equivalent to a call of the ``evaluate_by_index``
          method.

    See Also
    --------
    MeanAbsolutePercentageError

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import (
    ...     MeanAbsolutePercentageErrorStabilized,
    ... )
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2.5, 4, 2, 8])
    >>> metric = MeanAbsolutePercentageErrorStabilized()
    >>> metric(y_true, y_pred)
    np.float64(0.13004235907461714)
    """

    _tags = {
        "authors": ["michaelellis003"],
    }

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
        eps=None,
    ):
        self.eps = eps
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from
        evaluate_by_index

        Parameters
        ----------
        y_true : pandas.DataFrame with RangeIndex, integer index, or
            DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_pred : pandas.DataFrame with RangeIndex, integer index, or
            DatetimeIndex
            Predicted values to evaluate.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.

            * pd.Series if self.multioutput="uniform_average" or array-like;
              index is equal to index of y_true;
              entry at index i is metric at time i, averaged over variables.
            * pd.DataFrame if self.multioutput="raw_values";
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j.
        """
        multioutput = self.multioutput

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        # compute rolling MAD S_i for each column of y_true
        y_true_vals = y_true.values.astype(np.float64)
        n = y_true_vals.shape[0]

        # precompute cumulative means to avoid redundant slicing
        cumsum = np.cumsum(y_true_vals, axis=0)
        counts = np.arange(1, n + 1, dtype=np.float64)
        if y_true_vals.ndim > 1:
            counts = counts.reshape(-1, 1)
        rolling_mean = cumsum / counts

        s_vals = np.zeros_like(y_true_vals, dtype=np.float64)
        for i in range(1, n):
            s_vals[i] = np.abs(
                y_true_vals[:i] - rolling_mean[i - 1]
            ).mean(axis=0)

        numerator = (y_true - y_pred).abs()
        denominator_vals = (np.abs(y_true.values) + np.abs(y_pred.values)) / 2.0
        denominator_vals = denominator_vals + s_vals
        denominator_vals = np.maximum(denominator_vals, eps)

        raw_values = pd.DataFrame(
            numerator.values / denominator_vals,
            index=y_true.index,
            columns=y_true.columns,
        )

        raw_values = self._get_weighted_df(raw_values, **kwargs)

        return self._handle_multioutput(raw_values, multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or ``MyClass(**params[i])``
            creates a valid test instance.
            ``create_test_instance`` uses the first (or only) dictionary in
            ``params``
        """
        params1 = {}
        params2 = {"eps": 1e-6}
        return [params1, params2]
