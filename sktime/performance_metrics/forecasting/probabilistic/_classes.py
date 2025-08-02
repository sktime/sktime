#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from logging import warning

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils import check_array, check_consistent_length

from sktime.datatypes import check_is_scitype, convert, convert_to
from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.performance_metrics.forecasting._coerce import _coerce_to_scalar

# TODO: Rework tests now


class _BaseProbaForecastingErrorMetric(BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    score_average : bool, optional, default=True
        for interval and quantile losses only
            if True, metric/loss is averaged by upper/lower and/or quantile
            if False, metric/loss is not averaged by upper/lower and/or quantile
    """

    _tags = {
        "object_type": ["metric_forecasting_proba", "metric"],
        "reserved_params": ["multioutput", "score_average"],
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.multioutput = multioutput
        self.score_average = score_average
        super().__init__(multioutput=multioutput)

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic prediction method scitype:y_pred
            must be at fh and for variables equal to those in y_true.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        return self.evaluate(y_true, y_pred, multioutput=self.multioutput, **kwargs)

    def evaluate(self, y_true, y_pred, multioutput=None, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic prediction method scitype:y_pred
            must be at fh and for variables equal to those in y_true

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warning(
                "Dropping 0 width interval, don't include 0.5 quantile\
            for interval metrics."
            )

        # pass to inner function
        out = self._evaluate(y_true_inner, y_pred_inner, multioutput, **kwargs)

        if isinstance(multioutput, str):
            if self.score_average and multioutput == "uniform_average":
                out = out.mean(axis=1).iloc[0]  # average over all
            if self.score_average and multioutput == "raw_values":
                out = out.T.groupby(level=0).mean().T  # average over scores
            if not self.score_average and multioutput == "uniform_average":
                out = out.T.groupby(level=1).mean().T  # average over variables
            if not self.score_average and multioutput == "raw_values":
                out = out  # don't average
        else:  # is np.array with weights
            if self.score_average:
                out_raw = out.T.groupby(level=0).mean().T
                out = out_raw.dot(multioutput)[0]
            else:
                out = _groupby_dot(out, multioutput)

        if isinstance(out, pd.DataFrame):
            out = out.squeeze(axis=0)

        return out

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.DataFrame or of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.DataFrame of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : pd.DataFrame of shape (, n_outputs), calculated loss metric.
        """
        # Default implementation relies on implementation of evaluate_by_index
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, multioutput)
            out_df = pd.DataFrame(index_df.mean(axis=0)).T
            out_df.columns = index_df.columns
            return out_df
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def evaluate_by_index(self, y_true, y_pred, multioutput=None, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic prediction method scitype:y_pred
            must be at fh and for variables equal to those in y_true

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : pd.DataFrame of length len(fh), with calculated metric value(s)
            i-th column contains metric value(s) for prediction at i-th fh element
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warning(
                "Dropping 0 width interval, don't include 0.5 quantile\
            for interval metrics."
            )

        # pass to inner function
        out = self._evaluate_by_index(y_true_inner, y_pred_inner, multioutput, **kwargs)

        if isinstance(multioutput, str):
            if self.score_average and multioutput == "uniform_average":
                out = out.mean(axis=1)  # average over all
            if self.score_average and multioutput == "raw_values":
                out = out.T.groupby(level=0).mean().T  # average over scores
            if not self.score_average and multioutput == "uniform_average":
                out = out.T.groupby(level=1).mean().T  # average over variables
            if not self.score_average and multioutput == "raw_values":
                out = out  # don't average
        else:  # numpy array
            if self.score_average:
                out_raw = out.T.groupby(level=0).mean().T
                out = out_raw.dot(multioutput)
            else:
                out = _groupby_dot(out, multioutput)

        return out

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        By default this uses _evaluate to find jackknifed pseudosamples. This
        estimates the error at each of the time points.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        n = y_true.shape[0]
        out_series = pd.Series(index=y_pred.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
            for i in range(n):
                out_series[i] = n * x_bar - (n - 1) * self.evaluate(
                    np.vstack((y_true[:i, :], y_true[i + 1 :, :])),
                    np.vstack((y_pred[:i, :], y_pred[i + 1 :, :])),
                    multioutput,
                    **kwargs,
                )
            return out_series
        except RecursionError:
            raise RecursionError(
                "Must implement one of _evaluate or _evaluate_by_index"
            )

    def _check_consistent_input(self, y_true, y_pred, multioutput):
        check_consistent_length(y_true, y_pred)

        y_true = check_array(y_true, ensure_2d=False)

        if not isinstance(y_pred, pd.DataFrame):
            raise ValueError("y_pred should be a dataframe.")

        if not np.all([is_numeric_dtype(y_pred[c]) for c in y_pred.columns]):
            raise ValueError("Data should be numeric.")

        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))

        n_outputs = y_true.shape[1]

        allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    "Allowed 'multioutput' string values are "
                    f"{allowed_multioutput_str}. "
                    f"You provided multioutput={multioutput!r}"
                )
        elif multioutput is not None:
            multioutput = check_array(multioutput, ensure_2d=False)
            if n_outputs == 1:
                raise ValueError("Custom weights are useful only in multi-output case.")
            elif n_outputs != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), n_outputs)
                )

        return y_true, y_pred, multioutput

    def _check_ys(self, y_true, y_pred, multioutput):
        if multioutput is None:
            multioutput = self.multioutput
        valid, msg, metadata = check_is_scitype(
            y_pred, scitype="Proba", return_metadata=True, var_name="y_pred"
        )

        if not valid:
            raise TypeError(msg)

        y_pred_mtype = metadata["mtype"]
        inner_y_pred_mtype = self.get_tag("scitype:y_pred")

        y_pred_inner = convert(
            y_pred,
            from_type=y_pred_mtype,
            to_type=inner_y_pred_mtype,
            as_scitype="Proba",
        )

        if inner_y_pred_mtype == "pred_interval":
            if 0.0 in y_pred_inner.columns.get_level_values(1):
                for var in y_pred_inner.columns.get_level_values(0):
                    y_pred_inner[var, 0.0, "upper"] = y_pred_inner[var, 0.0, "lower"]

        y_pred_inner.sort_index(axis=1, level=[0, 1], inplace=True)

        y_true, y_pred, multioutput = self._check_consistent_input(
            y_true, y_pred, multioutput
        )

        return y_true, y_pred_inner, multioutput

    def _get_alpha_from(self, y_pred):
        """Fetch the alphas present in y_pred."""
        alphas = np.unique(list(y_pred.columns.get_level_values(1)))
        if not all((alphas > 0) & (alphas < 1)):
            raise ValueError("Alpha must be between 0 and 1.")

        return alphas

    def _check_alpha(self, alpha):
        """Check alpha input and coerce to np.ndarray."""
        if alpha is None:
            return None

        if isinstance(alpha, float):
            alpha = [alpha]

        if not isinstance(alpha, np.ndarray):
            alpha = np.asarray(alpha)

        if not all((alpha > 0) & (alpha < 1)):
            raise ValueError("Alpha must be between 0 and 1.")

        return alpha

    def _check_coverage(self, coverage):
        """Check coverage input and coerce to np.ndarray."""
        if coverage is None:
            return None

        if isinstance(coverage, float):
            coverage = [coverage]

        if not isinstance(coverage, np.ndarray):
            coverage = np.asarray(coverage)

        if not all((coverage > 0) & (coverage < 1)):
            raise ValueError("Coverage must be between 0 and 1.")

        return coverage

    def _handle_multioutput(self, loss, multioutput):
        """Handle output according to multioutput parameter.

        Parameters
        ----------
        loss : float, np.ndarray the evaluated metric value.

        multioutput : string "uniform_average" or "raw_values" determines how
            multioutput results will be treated.
        """
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return loss
            elif multioutput == "uniform_average":
                # pass None as weights to np.average: uniform mean
                multioutput = None
            else:
                raise ValueError(
                    "multioutput is expected to be 'raw_values' "
                    "or 'uniform_average' but we got %r"
                    " instead." % multioutput
                )

        if loss.ndim > 1:
            out = np.average(loss, weights=multioutput, axis=1)
        else:
            out = np.average(loss, weights=multioutput)
        return out


def _groupby_dot(df, weights):
    """Groupby dot product.

    Groups df by axis 1, level 1, and applies dot product with weights.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to groupby
    weights : np.array
        weights to apply to each group

    Returns
    -------
    out : pd.DataFrame
        dataframe with weighted groupby dot product
    """
    out = df.T.groupby(level=1).apply(lambda x: x.T.dot(weights)).T
    # formerly
    # out = df.groupby(axis=1, level=1).apply(lambda x: x.dot(weights))
    # but groupby(axis=1) is deprecated
    return out


class PinballLoss(_BaseProbaForecastingErrorMetric):
    r"""Pinball loss aka quantile loss for quantile/interval predictions.

    Can be used for both quantile and interval predictions.

    For a quantile prediction :math:`\widehat{y}
    at quantile point :math:`\alpha`,
    and a ground truth value :math:`y`, the pinball loss is defined as
    :math:`L_\alpha(y, \widehat{y}) := (y - \widehat{y}) \cdot (\alpha - H(y - \widehat{y}))`,
    where :math:`H` is the Heaviside step function defined as
    :math:`H(x) = 1` if :math:`x \ge 0` and :math:`H(x) = 0` otherwise.

    For a symmetric prediction interval :math:`I = [\widehat{y}_{\alpha}, \widehat{y}_{1 - \alpha}]`,
    the pinball loss is defined as
    :math:`L_\alpha(y, I) := L_\alpha(y, \widehat{y}_{\alpha}) + L_{1 - \alpha}(y, \widehat{y}_{1 - \alpha})`,
    or, in terms of coverage :math:`c = 1 - 2\alpha`, as
    :math:`L_c(y, I) := L_{1/2 - c/2}(y, a) + L_{1/2 + c/2}(y, b)`,
    if we write :math:`I = [a, b]`.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.
    * ``score_average`` controls averaging over quantiles/intervals.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.

        * If True, metric/loss is averaged over all quantiles present in ``y_pred``.
        * If False, metric/loss is not averaged over quantiles.

    alpha (optional) : float, list of float, or 1D array-like, default=None
        quantiles to evaluate metric at.
        Can be specified if no explicit quantiles are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
    >>> y_true = pd.Series([3, -0.5, 2, 7, 2])
    >>> y_pred = pd.DataFrame({
    ...     ('Quantiles', 0.05): [1.25, 0, 1, 4, 0.625],
    ...     ('Quantiles', 0.5): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles', 0.95): [3.75, 0, 3, 12, 1.875],
    ... })
    >>> pl = PinballLoss()
    >>> pl(y_true, y_pred)
    np.float64(0.1791666666666667)
    >>> pl = PinballLoss(score_average=False)
    >>> pl(y_true, y_pred).to_numpy()
    array([0.16625, 0.275  , 0.09625])
    >>> y_true = pd.DataFrame({
    ...     "Quantiles1": [3, -0.5, 2, 7, 2],
    ...     "Quantiles2": [4, 0.5, 3, 8, 3],
    ... })
    >>> y_pred = pd.DataFrame({
    ...     ('Quantiles1', 0.05): [1.5, -1, 1, 4, 0.65],
    ...     ('Quantiles1', 0.5): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles1', 0.95): [3.5, 4, 3, 12, 1.85],
    ...     ('Quantiles2', 0.05): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles2', 0.5): [5.0, 1, 4, 16, 2.5],
    ...     ('Quantiles2', 0.95): [7.5, 2, 6, 24, 3.75],
    ... })
    >>> pl = PinballLoss(multioutput='raw_values')
    >>> pl(y_true, y_pred).to_numpy()
    array([0.16233333, 0.465     ])
    >>> pl = PinballLoss(multioutput=np.array([0.3, 0.7]))
    >>> pl(y_true, y_pred)
    np.float64(0.3742000000000001)
    """  # noqa: E501

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True, alpha=None):
        self.score_average = score_average
        self.alpha = alpha
        self._alpha = self._check_alpha(alpha)
        self.metric_args = {"alpha": self._alpha}
        super().__init__(multioutput=multioutput, score_average=score_average)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target value`s.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values"
            Determines how multioutput results will be treated.
        """
        alpha = self._alpha
        y_pred_alphas = self._get_alpha_from(y_pred)
        if alpha is None:
            alphas = y_pred_alphas
        else:
            # if alpha was provided, check whether  they are predicted
            #   if not all alpha are observed, raise a ValueError
            if not np.isin(alpha, y_pred_alphas).all():
                # todo: make error msg more informative
                #   which alphas are missing
                msg = "not all quantile values in alpha are available in y_pred"
                raise ValueError(msg)
            else:
                alphas = alpha

        alphas = self._check_alpha(alphas)

        alpha_preds = y_pred.iloc[
            :, [x in alphas for x in y_pred.columns.get_level_values(1)]
        ]

        alpha_preds_np = alpha_preds.to_numpy()
        alpha_mat = np.repeat(
            (alpha_preds.columns.get_level_values(1).to_numpy().reshape(1, -1)),
            repeats=y_true.shape[0],
            axis=0,
        )

        y_true_np = np.repeat(y_true, axis=1, repeats=len(alphas))
        diff = y_true_np - alpha_preds_np
        sign = (diff >= 0).astype(diff.dtype)
        loss = alpha_mat * sign * diff - (1 - alpha_mat) * (1 - sign) * diff

        out_df = pd.DataFrame(loss, columns=alpha_preds.columns)

        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"alpha": [0.1, 0.5, 0.9]}
        return [params1, params2]


class EmpiricalCoverage(_BaseProbaForecastingErrorMetric):
    r"""Empirical coverage percentage for interval predictions.

    Applies to interval predictions.

    Should be used together with ``ConstraintViolation`` if reported.

    Up to a constant, ``PinballLoss`` is a weighted sum of ``ConstraintViolation`` and
    ``EmpiricalCoverage``.

    For an interval prediction :math:`I = [a, b]` and a ground truth value :math:`y`,
    the empirical coverage loss is defined as

    :math:`L(y, I) := 1, \text{if } y \in I, 0 \text{ otherwise}`.

    When averaged over test samples, variables, or coverages, the average
    is the same as the empirical coverage percentage, i.e., the percentage of
    predictions that contain the true value, among the values averaged over.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.
    * ``score_average`` controls averaging over quantiles/intervals.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    score_average : bool, optional, default = True
        specifies whether scores for each coverage value should be averaged.

        * If True, metric/loss is averaged over all coverages present in ``y_pred``.
        * If False, metric/loss is not averaged over coverages.

    coverage (optional) : float, list of float, or 1D array-like, default=None
        nominal coverage to evaluate metric at.
        Can be specified if no explicit coverages are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": False,
    }

    def __init__(
        self, multioutput="uniform_average", score_average=True, coverage=None
    ):
        self.score_average = score_average
        self.multioutput = multioutput
        self.coverage = coverage
        self._coverage = self._check_coverage(coverage)
        self.metric_args = {"coverage": self._coverage}
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true
        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        truth_array = (y_true_np >= lower).astype(int) * (y_true_np <= upper).astype(
            int
        )

        out_df = pd.DataFrame(
            truth_array, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"coverage": 0.5}
        return [params1, params2]


class IntervalWidth(_BaseProbaForecastingErrorMetric):
    """Interval width for interval predictions, sometimes also known as sharpness.

    Applies to interval predictions.

    Also known as "sharpness".

    For an interval prediction :math:`I = [a, b]` and a ground truth value :math:`y`,
    the interval width is defined as

    :math:`W(y, I) := b - a`.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.
    * ``score_average`` controls averaging over quantiles/intervals.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    score_average : bool, optional, default = True
        specifies whether scores for each coverage value should be averaged.

        * If True, metric/loss is averaged over all coverages present in ``y_pred``.
        * If False, metric/loss is not averaged over coverages.

    coverage (optional) : float, list of float, or 1D array-like, default=None
        nominal coverage to evaluate metric at.
        Can be specified if no explicit coverages are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": True,
    }

    def __init__(
        self, multioutput="uniform_average", score_average=True, coverage=None
    ):
        self.score_average = score_average
        self.multioutput = multioutput
        self.coverage = coverage
        self._coverage = self._check_coverage(coverage)
        self.metric_args = {"coverage": self._coverage}
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true
        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        vars = np.unique(y_pred.columns.get_level_values(0))

        metric_array = upper - lower

        out_df = pd.DataFrame(
            metric_array, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"coverage": 0.5}
        return [params1, params2]


class ConstraintViolation(_BaseProbaForecastingErrorMetric):
    r"""Average absolute constraint violations for interval predictions.

    Applies to interval predictions.

    Should be used together with ``EmpiricalCoverage`` if reported.

    Up to a constant, ``PinballLoss`` is a weighted sum of ``ConstraintViolation`` and
    ``EmpiricalCoverage``.

    For an interval prediction :math:`I = [a, b]` and a ground truth value :math:`y`,
    the constraint violation loss is defined as

    .. math::

        L(y, I) :=
        \begin{cases}
        a - y, & \text{if } y < a \\
        y - b, & \text{if } y > b \\
        0, & \text{otherwise}
        \end{cases}

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.
    * ``score_average`` controls averaging over quantiles/intervals.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    score_average : bool, optional, default = True
        specifies whether scores for each coverage value should be averaged.

        * If True, metric/loss is averaged over all coverages present in ``y_pred``.
        * If False, metric/loss is not averaged over coverages.

    coverage (optional) : float, list of float, or 1D array-like, default=None
        nominal coverage to evaluate metric at.
        Can be specified if no explicit coverages are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": True,
    }

    def __init__(
        self, multioutput="uniform_average", score_average=True, coverage=None
    ):
        self.score_average = score_average
        self.multioutput = multioutput
        self.coverage = coverage
        self._coverage = self._check_coverage(coverage)
        self.metric_args = {"coverage": self._coverage}
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true

        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        int_distance = ((y_true_np < lower).astype(int) * abs(lower - y_true_np)) + (
            (y_true_np > upper).astype(int) * abs(y_true_np - upper)
        )

        out_df = pd.DataFrame(
            int_distance, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"coverage": 0.5}
        return [params1, params2]


PANDAS_DF_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


class _BaseDistrForecastingMetric(_BaseProbaForecastingErrorMetric):
    """Intermediate base class for distributional prediction metrics/scores.

    Developer note:
    Experimental and overrides public methods of _BaseProbaForecastingErrorMetric.
    This should be refactored into one base class.
    """

    _tags = {
        "scitype:y_pred": "pred_proba",
        "lower_is_better": True,
    }

    def evaluate(self, y_true, y_pred, multioutput=None, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic prediction method scitype:y_pred
            must be at fh and for variables equal to those in y_true

        multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
                (n_outputs,), default='uniform_average'
            Defines whether and how to aggregate metric for across variables.

            * If 'uniform_average' (default), errors are mean-averaged across variables.
            * If array-like, errors are weighted averaged across variables,
            values as weights.
            * If 'raw_values', does not average errors across variables,
            columns are retained.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            float if multioutput = "uniform_average"
            metric is always averaged (arithmetic) over fh values
        """
        index_df = self.evaluate_by_index(y_true, y_pred, multioutput)
        out_df = pd.DataFrame(index_df.mean(axis=0)).T
        out_df.columns = index_df.columns

        if multioutput == "uniform_average":
            out_df = _coerce_to_scalar(out_df)
        return out_df

    def evaluate_by_index(
        self, y_true, y_pred, multioutput="uniform_average", **kwargs
    ):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : sktime BaseDistribution of same shape as y_true
            Predictive distribution.
            Must have same index and columns as y_true.

        Returns
        -------
        loss : ``pd.Series`` or ``pd.DataFrame``
            Calculated metric, by time point (default=jackknife pseudo-values).

            ``pd.Series`` if ``self.multioutput="uniform_average"`` or array-like

            * index is equal to index of ``y_true``
            * entry at index i is metric at time i, averaged over variables

            ``pd.DataFrame`` if ``self.multioutput="raw_values"``

            * index and columns equal to those of ``y_true``
            * i,j-th entry is metric at time i, at variable j
        """
        multivariate = self.multivariate

        y_true = convert_to(y_true, to_type=PANDAS_DF_MTYPES)

        if multivariate:
            res = self._evaluate_by_index(
                y_true=y_true, y_pred=y_pred, multioutput=multioutput
            )
            res.columns = ["score"]
            return res
        else:
            res_by_col = []
            for col in y_pred.columns:
                y_pred_col = y_pred.loc[:, [col]]
                y_true_col = y_true.loc[:, [col]]
                res_for_col = self._evaluate_by_index(
                    y_true=y_true_col, y_pred=y_pred_col, multioutput=multioutput
                )
                res_for_col.columns = [col]
                res_by_col += [res_for_col]
            res = pd.concat(res_by_col, axis=1)

        return res


class LogLoss(_BaseDistrForecastingMetric):
    r"""Logarithmic loss for distributional predictions.

    For a predictive distribution :math:`d` with pdf :math:`p_d`
    and a ground truth value :math:`y`, the logarithmic loss is
    defined as :math:`L(y, d) := -\log p_d(y)`.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    multivariate : bool, optional, default=False

        * if True, behaves as multivariate log-loss:
          the log-loss is computed for entire row, results one score per row
        * if False, is univariate log-loss:
          the log-loss is computed per variable marginal, results in many scores per row
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        res = -y_pred.log_pdf(y_true)
        # replace this by multivariate log_pdf once distr implements
        # i.e., pass multivariate on to log_pdf
        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["density"])
        else:
            return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]


class SquaredDistrLoss(_BaseDistrForecastingMetric):
    r"""Squared loss for distributional predictions.

    Also known as:

    * continuous Brier loss
    * Gneiting loss
    * (mean) squared error/loss, i.e., confusingly named the same as the
      point prediction loss commonly known as the mean squared error

    For a predictive distribution :math:`d`
    and a ground truth value :math:`y`, the squared (distribution) loss is
    defined as :math:`L(y, d) := -2 p_d(y) + \|p_d\|^2`,
    where :math:`\|p_d\|^2` is the (function) L2-norm of :math:`p_d`.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    multivariate : bool, optional, default=False

        * if True, behaves as multivariate squared loss:
          the score is computed for entire row, results one score per row
        * if False, is univariate squared loss:
          the score is computed per variable marginal, results in many scores per row
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        res = -2 * y_pred.log_pdf(y_true) + y_pred.pdfnorm(a=2)
        # replace this by multivariate log_pdf once distr implements
        # i.e., pass multivariate on to log_pdf
        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["density"])
        else:
            return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]


class CRPS(_BaseDistrForecastingMetric):
    r"""Continuous rank probability score for distributional predictions.

    Also known as:

    * integrated squared loss (ISL)
    * integrated Brier loss (IBL)
    * energy loss

    For a predictive distribution :math:`d` and a ground truth value :math:`y`,
    the CRPS is defined as
    :math:`L(y, d) := \mathbb{E}_{Y \sim d}|Y-y| - \frac{1}{2} \mathbb{E}_{X,Y \sim d}|X-Y|`.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    multivariate : bool, optional, default=False

        * if True, behaves as multivariate CRPS:
          the score is computed for entire row, results one score per row
        * if False, is univariate CRPS:
          the score is computed per variable marginal, results in many scores per row
    """  # noqa: E501

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        # CRPS(d, y) = E_X,Y as d [abs(Y-y) - 0.5 abs(X-Y)]
        return y_pred.energy(y_true) - y_pred.energy() / 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]


class AUCalibration(_BaseDistrForecastingMetric):
    r"""Area under the calibration curve for distributional predictions.

    Computes the unsigned area between the calibration curve and the diagonal.

    The calibration curve is the cumulative curve of the sample of
    predictive cumulative distribution functions evaluated at the true values.

    Mathematically, let :math:`d_1, \dots, d_N` be the predictive distributions,
    let :math:`y_1, \dots, y_N` be the true values, and let :math:`F_i` be the
    cumulative distribution function of :math:`d_i`.

    Define the calibration sample as :math:`c_i := F_i(y_i)`, for
    :math:`i = 1, \dots, N`. For perfect predictions, the sample of :math:`c_i` will be
    uniformly distributed on [0, 1], and i.i.d. from that uniform distribution.

    Let :math:`c_{(i)}` be the :math:`i`-th order statistic of the sample of
    :math:`c_i`, i.e., the :math:`i`-th smallest value in the sample.

    The (unsigned) area under the calibration curve - or, more precisely,
    between the diagonal and the calibration curve - is defined as

    .. math:: \frac{1}{N} \sum_{i=1}^N \left| c_{(i)} - \frac{i}{N} \right|.

    * ``evaluate`` returns the unsigned area between the calibration curve
      and the diagonal, i.e., the above quantity.
    * ``evaluate_by_index`` returns, for the :math:`i`-th test sample, the value
      :math:`\left| c_i - \frac{r_i}{N} \right|`,
      where :math:`r_i` is the rank of :math:`c_i`
      in the sample of :math:`c_i`. In case of ties, tied ranks are averaged.
    * ``multivariate`` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    multivariate : bool, optional, default=False

        * if True, behaves as multivariate metric (sum of scores):
          the metric is computed for entire row, results one score per row
        * if False, is univariate metric, per variable:
          the metric is computed per variable marginal, results in many scores per row
    """  # noqa: E501

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        cdfs = y_pred.cdf(y_true)
        # using the average in case of ranks is fine
        # because the absolute sums in the metric average out
        cdfs_ranked = cdfs.rank(axis=0, method="average", pct=True)
        n = cdfs.shape[0]
        diagonal = np.arange(1, n + 1).reshape(-1, 1) / n

        res = (cdfs_ranked - diagonal).abs()

        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["score"])
        return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]
