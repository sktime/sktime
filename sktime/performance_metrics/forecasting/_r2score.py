#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

__author__ = ["krishnadevan"]

import numpy as np
from sklearn.metrics import r2_score as _r2_score

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class R2Score(BaseForecastingErrorMetric):
    r"""R-squared (coefficient of determination) score.

    Best possible score is 1.0 and it can be negative (because the model can
    be arbitrarily worse). A constant model that always predicts the expected
    value of y, disregarding the input features, would get a score of 0.0.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns the R2 score,
    :math:`1 - \frac{\sum_{i=1}^n (y_i - \widehat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}`,
    where :math:`\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i`.

    This is a score metric (higher is better), providing a sktime-compatible
    interface to ``sklearn.metrics.r2_score``.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    Parameters
    ----------
    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          scores of all outputs are averaged with uniform weight.
        * If 1D array-like, scores are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable scores
          are returned.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          scores are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average scores across levels, hierarchy is retained.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call to the metric object averages over time points,
          equivalent to a call of the ``evaluate`` method.
        * If ``True``, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

    See Also
    --------
    ExplainedVarianceScore

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import R2Score
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> r2 = R2Score()
    >>> r2(y_true, y_pred)
    np.float64(0.9486081370449679)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> r2(y_true, y_pred)
    np.float64(0.9382566585956416)
    >>> r2 = R2Score(multioutput='raw_values')
    >>> r2(y_true, y_pred)
    array([0.96543779, 0.91107553])
    """

    _tags = {
        "lower_is_better": False,
    }

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the R2 score on given inputs.

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.
        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        score : float or np.ndarray
            R2 score value(s).
        """
        multioutput = self.multioutput

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        sample_weight = self._set_sample_weight_on_kwargs(**kwargs)

        return _r2_score(
            y_true_np,
            y_pred_np,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"multioutput": "raw_values"}
        return [params1, params2]
