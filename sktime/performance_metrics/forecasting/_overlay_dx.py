#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Overlay DX metric for visual alignment quality assessment.

The overlay_dx metric measures "visual alignment" quality by computing
the percentage of predictions that fall within a tolerance threshold
of the true values across a range of tolerance values.
"""

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class OverlayDX(BaseForecastingErrorMetric):
    r"""Overlay DX: tolerance-sweep visual alignment metric.

    The overlay_dx metric measures visual alignment quality between predictions
    and actual values by computing the percentage of predictions within tolerance
    across a range of tolerance thresholds.

    This provides an interpretable "alignment quality" metric that is more robust
    than traditional point metrics like MAE or RMSE, as it considers how well
    predictions align with actual values across different tolerance levels.

    The metric computes:
    For each tolerance threshold t in the range [min_percentage, max_percentage]:
        - Calculate the percentage of predictions where |y_true - y_pred| <= t
        - Return the area under this curve (AUC) normalized by the range

    Parameters
    ----------
    multioutput : {'uniform_average', 'raw_values'}, or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        * 'raw_values' : Returns a full set of errors in case of multioutput input.
        * 'uniform_average' : Errors of all outputs are averaged with uniform weight.

    multilevel : {'uniform_average', 'uniform_average_time', 'raw_values'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights for computing weighted average.

    max_percentage : float, default=100.0
        Maximum tolerance threshold as a percentage of the value range.

    min_percentage : float, default=0.1
        Minimum tolerance threshold as a percentage of the value range.

    step : float, default=0.1
        Step size for tolerance threshold sweep.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call to the metric object averages over time points,
          equivalent to a call of the ``evaluate`` method.
        * If ``True``, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

    Attributes
    ----------
    name : str
        Name of the metric = "OverlayDX"

    greater_is_better : bool
        True, since higher overlay_dx indicates better visual alignment.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import OverlayDX
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> overlay_dx = OverlayDX()
    >>> overlay_dx(y_true, y_pred)  # doctest: +SKIP
    >>> # Returns a score between 0 and 100, higher is better

    References
    ----------
    Inspired by visual alignment metrics used in time series analysis.
    See: https://github.com/Smile-SA/overlay_dx
    """

    _tags = {
        "object_type": ["metric_forecasting", "metric"],
        "scitype:y_pred": "pred",
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "capability:multivariate": True,
        "lower_is_better": False,  # Higher is better for this metric
        "inner_implements_multilevel": False,
        "reserved_params": ["multioutput", "multilevel", "by_index"],
    }

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sample_weight=None,
        max_percentage=100.0,
        min_percentage=0.1,
        step=0.1,
        by_index=False,
    ):
        self.sample_weight = sample_weight
        self.max_percentage = max_percentage
        self.min_percentage = min_percentage
        self.step = step
        self.name = "OverlayDX"
        self.greater_is_better = True

        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.
        """
        # For by_index evaluation, we compute the metric at each time point
        # This returns whether each prediction is within tolerance
        multioutput = self.multioutput

        # Compute absolute errors
        abs_errors = (y_true - y_pred).abs()

        # Get the value range for normalization
        value_range = y_true.max() - y_true.min()
        if isinstance(value_range, pd.Series):
            value_range = value_range.values
        value_range = np.maximum(value_range, 1e-10)  # Avoid division by zero

        # For by_index, return normalized inverse error (higher is better)
        # This gives a per-timepoint quality score
        normalized_error = abs_errors / value_range
        quality_score = 100.0 * (1.0 - normalized_error.clip(upper=1.0))

        quality_score = self._get_weighted_df(quality_score, **kwargs)

        return self._handle_multioutput(quality_score, multioutput)

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the metric.

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        loss : float or pd.Series
            Calculated metric value.
        """
        # Convert to numpy for computation
        y_true_np = y_true.values if hasattr(y_true, "values") else y_true
        y_pred_np = y_pred.values if hasattr(y_pred, "values") else y_pred

        # Flatten if needed
        if y_true_np.ndim > 1:
            n_samples, n_outputs = y_true_np.shape
        else:
            n_samples = len(y_true_np)
            n_outputs = 1
            y_true_np = y_true_np.reshape(-1, 1)
            y_pred_np = y_pred_np.reshape(-1, 1)

        # Compute absolute errors
        abs_errors = np.abs(y_true_np - y_pred_np)

        # Get value range for each output
        value_ranges = np.ptp(y_true_np, axis=0)  # peak-to-peak (max - min)
        value_ranges = np.maximum(value_ranges, 1e-10)  # Avoid division by zero

        # Generate tolerance thresholds
        thresholds = np.arange(
            self.min_percentage,
            self.max_percentage + self.step,
            self.step
        )

        # Compute overlay scores for each output
        scores = []
        for output_idx in range(n_outputs):
            output_errors = abs_errors[:, output_idx]
            output_range = value_ranges[output_idx]

            # Compute percentage within tolerance for each threshold
            percentages_within = []
            for threshold_pct in thresholds:
                tolerance = (threshold_pct / 100.0) * output_range
                within_tolerance = np.sum(output_errors <= tolerance)
                pct_within = (within_tolerance / n_samples) * 100.0
                percentages_within.append(pct_within)

            # Compute area under curve (AUC) using trapezoidal rule
            try:
                from scipy.integrate import trapezoid
                auc = trapezoid(percentages_within, thresholds)
            except ImportError:
                # Manual trapezoidal integration for compatibility
                auc = 0.0
                for i in range(len(thresholds) - 1):
                    auc += (percentages_within[i] + percentages_within[i+1]) / 2.0 * (thresholds[i+1] - thresholds[i])
            # Normalize by the range of thresholds
            normalized_score = auc / (self.max_percentage - self.min_percentage)
            scores.append(normalized_score)

        scores = np.array(scores)

        # Handle multioutput
        if self.multioutput == "raw_values":
            if n_outputs == 1:
                return scores[0]
            else:
                return pd.Series(scores, index=y_true.columns if hasattr(y_true, "columns") else None)
        elif self.multioutput == "uniform_average":
            return np.mean(scores)
        elif isinstance(self.multioutput, (list, np.ndarray)):
            weights = np.array(self.multioutput)
            return np.average(scores, weights=weights)
        else:
            return np.mean(scores)


def overlay_dx_score(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    max_percentage=100.0,
    min_percentage=0.1,
    step=0.1,
):
    """Overlay DX score - tolerance-sweep visual alignment metric.

    Functional interface to the OverlayDX metric class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.

    max_percentage : float, default=100.0
        Maximum tolerance threshold as percentage.

    min_percentage : float, default=0.1
        Minimum tolerance threshold as percentage.

    step : float, default=0.1
        Step size for tolerance sweep.

    Returns
    -------
    score : float or ndarray of floats
        Overlay DX score. Higher is better.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import overlay_dx_score
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> overlay_dx_score(y_true, y_pred)  # doctest: +SKIP
    """
    metric = OverlayDX(
        multioutput=multioutput,
        sample_weight=sample_weight,
        max_percentage=max_percentage,
        min_percentage=min_percentage,
        step=step,
    )
    return metric(y_true, y_pred)
