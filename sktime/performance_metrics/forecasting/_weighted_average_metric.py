from sktime.performance_metrics.forecasting._classes import (
    BaseForecastingErrorMetric,
    MeanAbsoluteError,
    MeanSquaredError,
    MedianAbsoluteError,
)

__author__ = ["benheid"]
__all__ = [
    "WeightedAverageMetric",
]


class WeightedAverageMetric(BaseForecastingErrorMetric):
    def __init__(
        self,
        metrics=[],
        weights=[],
        normalize=True,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        """Initialize WeightedAverageMetric.

        WeightedAverageMetric is a wrapper for combining multiple
        forecasting error metrics into a single metric using weighted
        averaging.

        Parameters
        ----------
        metrics : list of BaseForecastingErrorMetric
            List of metrics to be used in the ensemble.
        weights : list of float
            List of weights for each metric in the ensemble.
            if None, all metrics are equally weighted.
        normalize : bool, default=True
            If True, the weights are normalized to sum to 1.
        multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
                (n_outputs,), default='uniform_average'
            Defines whether and how to aggregate metric for across variables.

            * If 'uniform_average' (default), errors are mean-averaged across variables.
            * If array-like, errors are weighted averaged across variables,
            values as weights.
            * If 'raw_values', does not average errors across variables,
            columns are retained.

        multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
            Defines how to aggregate metric for hierarchical data (with levels).

            * If 'uniform_average' (default), errors are mean-averaged across levels.
            * If 'uniform_average_time', metric is applied to all data,
            ignoring level index.
            * If 'raw_values', does not average errors across levels, hierarchy is
            retained.

        by_index : bool, default=False
            Determines averaging over time points in direct call to metric object.

            * If False, direct call to the metric object averages over time points,
            equivalent to a call of the``evaluate`` method.
            * If True, direct call to the metric object evaluates the metric at each
            time point, equivalent to a call of the ``evaluate_by_index`` method.

        Examples
        --------
        >>> from sktime.performance_metrics.forecasting import \
        WeightedAverageMetric, MeanAbsoluteError, MeanSquaredError
        >>> import numpy as np
        >>> y_true = np.array([3, -2, 2, 8, 2])
        >>> y_pred = np.array([1, 0.0, 2, 8, 2])
        >>> mae = MeanAbsoluteError()
        >>> mse = MeanSquaredError()
        >>> ensemble_metric = WeightedAverageMetric(
        ...     metrics=[mae, mse],
        ...     weights=[0.5, 0.5]
        ... )
        >>> ensemble_metric.evaluate(y_true, y_pred)
        np.float64(1.2)

        """
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )
        self.metrics = metrics
        self.weights = weights
        self.normalize = normalize

        for metric in self.metrics:
            metric.set_params(
                multioutput=multioutput,
                multilevel=multilevel,
                by_index=by_index,
            )

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the ensemble metric.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.
        **kwargs : dict
            Additional arguments to be passed to the individual metrics.

        Returns
        -------
        float
            The weighted ensemble metric value.
        """
        total = 0.0
        weights = self.weights if self.normalize else [1.0] * len(self.metrics)
        weights = weights / sum(weights) if self.normalize else weights
        for metric, weight in zip(self.metrics, weights):
            total += weight * metric.evaluate(y_true, y_pred, **kwargs)
        return total

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        By default this uses _evaluate to find jackknifed pseudosamples.
        This yields estimates for the metric at each of the time points.
        Caution: this is only sensible for differentiable statistics,
        i.e., not for medians, quantiles or median/quantile based statistics.

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
        total = 0.0
        weights = self.weights if self.normalize else [1.0] * len(self.metrics)
        weights = weights / sum(weights) if self.normalize else weights
        for metric, weight in zip(self.metrics, weights):
            total += weight * metric.evaluate_by_index(y_true, y_pred, **kwargs)
        return total

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {
                "metrics": [
                    MeanAbsoluteError(),
                    MeanSquaredError(),
                    MedianAbsoluteError(),
                ],
                "weights": [0.5, 0.3, 0.2],
            },
            {
                "metrics": [
                    MeanAbsoluteError(),
                ],
                "weights": [0.4],
            },
        ]
