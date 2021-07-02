# -*- coding: utf-8 -*-
from sktime.performance_metrics.base import BaseMetric
from sktime.performance_metrics.forecasting._functions import (
    relative_loss,
    mean_asymmetric_error,
    mean_absolute_scaled_error,
    median_absolute_scaled_error,
    mean_squared_scaled_error,
    median_squared_scaled_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    median_squared_error,
    mean_absolute_percentage_error,
    median_absolute_percentage_error,
    mean_squared_percentage_error,
    median_squared_percentage_error,
    mean_relative_absolute_error,
    median_relative_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
)

__author__ = ["Markus Löning", "Tomasz Chodakowski", "Ryan Kuhns"]
__all__ = [
    "make_forecasting_scorer",
    "MeanAbsoluteScaledError",
    "MedianAbsoluteScaledError",
    "MeanSquaredScaledError",
    "MedianSquaredScaledError",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "MedianAbsoluteError",
    "MedianSquaredError",
    "MeanAbsolutePercentageError",
    "MedianAbsolutePercentageError",
    "MeanSquaredPercentageError",
    "MedianSquaredPercentageError",
    "MeanRelativeAbsoluteError",
    "MedianRelativeAbsoluteError",
    "GeometricMeanRelativeAbsoluteError",
    "GeometricMeanRelativeSquaredError",
    "MeanAsymmetricError",
    "RelativeLoss",
]


class _BaseForecastingErrorMetric(BaseMetric):
    """Base class for defining forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
    }

    greater_is_better = False

    def __init__(self, func, name=None, multioutput="uniform_average"):
        self.multioutput = multioutput
        super().__init__(func, name=name)

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
                (n_timepoints, n_outputs), default = None
            Optional keyword argument to pass training data.

        y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) \
             or (fh, n_outputs) where fh is the forecasting horizon
            Optional keyword argument to pass benchmark predictions.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(y_true, y_pred, multioutput=self.multioutput, **kwargs)


class _BaseForecastingScoreMetric(_BaseForecastingErrorMetric):
    """Base class for defining forecasting score metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting score
    metrics measure the agreement between forecasts and true values. Higher
    values are better.
    """

    greater_is_better = True


class _PercentageErrorMixin:
    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Uses `symmetric` attribute to determine whether underlying function
        should return symmetric percentage error metric or a percentage error
        metric.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
                (n_timepoints, n_outputs), default = None
            Optional keyword argument to pass training data.

        y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) \
             or (fh, n_outputs) where fh is the forecasting horizon
            Optional keyword argument to pass benchmark predictions.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(
            y_true,
            y_pred,
            multioutput=self.multioutput,
            symmetric=self.symmetric,
            **kwargs,
        )


class _SquaredErrorMixin:
    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Uses `square_root` attribute to determine whether the
        underlying function should return the square_root of the metric or
        the metric.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
                (n_timepoints, n_outputs), default = None
            Optional keyword argument to pass training data.

        y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) \
             or (fh, n_outputs) where fh is the forecasting horizon
            Optional keyword argument to pass benchmark predictions.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(
            y_true,
            y_pred,
            multioutput=self.multioutput,
            square_root=self.square_root,
            **kwargs,
        )


class _SquaredPercentageErrorMixin:
    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Uses `symmetric` attribute to determine whether underlying function
        should return symmetric percentage error metric or a percentage error
        metric. Also uses `square_root` attribute to determine whether the
        underlying function should return the square_root of the metric or
        the metric.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
                (n_timepoints, n_outputs), default = None
            Optional keyword argument to pass training data.

        y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) \
             or (fh, n_outputs) where fh is the forecasting horizon
            Optional keyword argument to pass benchmark predictions.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(
            y_true,
            y_pred,
            multioutput=self.multioutput,
            symmetric=self.symmetric,
            square_root=self.square_root,
            **kwargs,
        )


class _AsymmetricErrorMixin:
    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
                (n_timepoints, n_outputs), default = None
            Optional keyword argument to pass training data.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(
            y_true,
            y_pred,
            multioutput=self.multioutput,
            asymmetric_threshold=self.asymmetric_threshold,
            left_error_function=self.left_error_function,
            right_error_function=self.right_error_function,
            **kwargs,
        )


class _RelativeLossMixin:
    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) \
             or (fh, n_outputs) where fh is the forecasting horizon
            Optional keyword argument to pass benchmark predictions.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(
            y_true,
            y_pred,
            multioutput=self.multioutput,
            relative_loss_function=self.relative_loss_function,
            **kwargs,
        )


class _ScaledForecastingErrorMetric(_BaseForecastingErrorMetric):
    """Base class for defining forecasting success metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting success
    metrics measure the agreement between forecasts and true values. Higher
    values are better.
    """

    _tags = {
        "requires-y-train": True,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
    }

    def __init__(self, func, name=None, multioutput="uniform_average", sp=1):
        self.sp = sp
        super().__init__(func=func, name=name, multioutput=multioutput)


class _ScaledSquaredForecastingErrorMetric(
    _SquaredErrorMixin, _ScaledForecastingErrorMetric
):
    def __init__(
        self, func, name=None, multioutput="uniform_average", sp=1, square_root=False
    ):
        self.square_root = square_root
        super().__init__(func=func, name=name, multioutput=multioutput, sp=sp)


class _PercentageForecastingErrorMetric(
    _PercentageErrorMixin, _BaseForecastingErrorMetric
):
    def __init__(self, func, name=None, multioutput="uniform_average", symmetric=True):
        self.symmetric = symmetric
        super().__init__(func=func, name=name, multioutput=multioutput)


class _SquaredForecastingErrorMetric(_SquaredErrorMixin, _BaseForecastingErrorMetric):
    def __init__(
        self, func, name=None, multioutput="uniform_average", square_root=False
    ):
        self.square_root = square_root
        super().__init__(func=func, name=name, multioutput=multioutput)


class _SquaredPercentageForecastingErrorMetric(
    _SquaredPercentageErrorMixin, _BaseForecastingErrorMetric
):
    def __init__(
        self,
        func,
        name=None,
        multioutput="uniform_average",
        square_root=False,
        symmetric=True,
    ):
        self.square_root = square_root
        self.symmetric = symmetric
        super().__init__(func=func, name=name, multioutput=multioutput)


class _AsymmetricForecastingErrorMetric(
    _AsymmetricErrorMixin, _BaseForecastingErrorMetric
):
    def __init__(
        self,
        func,
        name=None,
        multioutput="uniform_average",
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
    ):
        self.asymmetric_threshold = asymmetric_threshold
        self.left_error_function = left_error_function
        self.right_error_function = right_error_function
        super().__init__(func=func, name=name, multioutput=multioutput)


class _RelativeLossForecastingErrorMetric(
    _RelativeLossMixin, _BaseForecastingErrorMetric
):
    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    def __init__(
        self,
        func,
        name=None,
        multioutput="uniform_average",
        relative_loss_function=mean_absolute_error,
    ):
        self.relative_loss_function = relative_loss_function
        super().__init__(func=func, name=name, multioutput=multioutput)


def make_forecasting_scorer(
    func, name=None, greater_is_better=False, multioutput="uniform_average"
):
    """Create a metric class from metric functions.

    Parameters
    ----------
    func
        Function to convert to a forecasting scorer class.
        Score function (or loss function) with signature ``func(y, y_pred, **kwargs)``.

    name : str, default=None
        Name to use for the forecasting scorer loss class.

    greater_is_better : bool, default=False
        If True then maximizing the metric is better.
        If False then minimizing the metric is better.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    scorer:
        Metric class that can be used as forecasting scorer.
    """
    if greater_is_better:
        return _BaseForecastingErrorMetric(func, name=name, multioutput=multioutput)
    else:
        return _BaseForecastingScoreMetric(func, name=name, multioutput=multioutput)


class MeanAbsoluteScaledError(_ScaledForecastingErrorMetric):
    """Mean absolute scaled error (MASE).

    MASE output is non-negative floating point. The best value is 0.0.

    Like other scaled performance metrics, this scale-free error metric can be
    used to compare forecast methods on a single series and also to compare
    forecast accuracy between series.

    This metric is well suited to intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of the data

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    sp : int
        Stores seasonal periodicity of data.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MedianAbsoluteScaledError
    MeanSquaredScaledError
    MedianSquaredScaledError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mase = MeanAbsoluteScaledError()
    >>> mase(y_true, y_pred, y_train=y_train)
    0.18333333333333335
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mase(y_true, y_pred, y_train=y_train)
    0.18181818181818182
    >>> mase = MeanAbsoluteScaledError(multioutput='raw_values')
    >>> mase(y_true, y_pred, y_train=y_train)
    array([0.10526316, 0.28571429])
    >>> mase = MeanAbsoluteScaledError(multioutput=[0.3, 0.7])
    >>> mase(y_true, y_pred, y_train=y_train)
    0.21935483870967742

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
    for intermittent demand", Foresight, Issue 4.

    Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
    "The M4 Competition: 100,000 time series and 61 forecasting methods",
    International Journal of Forecasting, Volume 3.
    """

    def __init__(self, multioutput="uniform_average", sp=1):
        name = "MeanAbsoluteScaledError"
        func = mean_absolute_scaled_error
        super().__init__(func=func, name=name, multioutput=multioutput, sp=sp)


class MedianAbsoluteScaledError(_ScaledForecastingErrorMetric):
    """Median absolute scaled error (MdASE).

    MdASE output is non-negative floating point. The best value is 0.0.

    Taking the median instead of the mean of the test and train absolute errors
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Like MASE and other scaled performance metrics this scale-free metric can be
    used to compare forecast methods on a single series or between series.

    Also like MASE, this metric is well suited to intermittent-demand series
    because it will not give infinite or undefined values unless the training
    data is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of data.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    sp : int
        Stores seasonal periodicity of data.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsoluteScaledError
    MeanSquaredScaledError
    MedianSquaredScaledError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianAbsoluteScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mdase = MedianAbsoluteScaledError()
    >>> mdase(y_true, y_pred, y_train=y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdase(y_true, y_pred, y_train=y_train)
    0.18181818181818182
    >>> mdase = MedianAbsoluteScaledError(multioutput='raw_values')
    >>> mdase(y_true, y_pred, y_train=y_train)
    array([0.10526316, 0.28571429])
    >>> mdase = MedianAbsoluteScaledError(multioutput=[0.3, 0.7])
    >>> mdase( y_true, y_pred, y_train=y_train)
    0.21935483870967742

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
    for intermittent demand", Foresight, Issue 4.

    Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
    "The M4 Competition: 100,000 time series and 61 forecasting methods",
    International Journal of Forecasting, Volume 3.
    """

    def __init__(self, multioutput="uniform_average", sp=1):
        name = "MedianAbsoluteScaledError"
        func = median_absolute_scaled_error
        super().__init__(func=func, name=name, multioutput=multioutput, sp=sp)


class MeanSquaredScaledError(_ScaledSquaredForecastingErrorMetric):
    """Mean squared scaled error (MSSE) or root mean squared scaled error (RMSSE).

    If `square_root` is False then calculates MSSE, otherwise calculates RMSSE if
    `square_root` is True. Both MSSE and RMSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared varient of the MASE loss metric.  Like MASE and other
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

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    sp : int
        Stores seasonal periodicity of data.

    square_root : bool
        Stores whether to take the square root of the metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MedianSquaredScaledError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanSquaredScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> rmsse = MeanSquaredScaledError(square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    0.20568833780186058
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> rmsse(y_true, y_pred, y_train=y_train)
    0.15679361328058636
    >>> rmsse = MeanSquaredScaledError(multioutput='raw_values', square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    array([0.11215443, 0.20203051])
    >>> rmsse = MeanSquaredScaledError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    0.17451891814894502

    References
    ----------
    M5 Competition Guidelines.
    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average", sp=1, square_root=False):
        name = "MeanSquaredScaledError"
        func = mean_squared_scaled_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            sp=1,
            square_root=square_root,
        )


class MedianSquaredScaledError(_ScaledSquaredForecastingErrorMetric):
    """Median squared scaled error (MdSSE) or root median squared scaled error (RMdSSE).

    If `square_root` is False then calculates MdSSE, otherwise calculates RMdSSE if
    `square_root` is True. Both MdSSE and RMdSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared varient of the MdASE loss metric. Like MASE and other
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
    sp : int
        Seasonal periodicity of data.

    square_root : bool, default = False
        Whether to take the square root of the metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    sp : int
        Seasonal periodicity of data.

    square_root : bool
        Stores whether to take the square root of the metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MedianSquaredScaledError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianSquaredScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> rmdsse = MedianSquaredScaledError(square_root=True)
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    0.1472819539849714
    >>> rmdsse = MedianSquaredScaledError(multioutput='raw_values', square_root=True)
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    array([0.08687445, 0.20203051])
    >>> rmdsse = MedianSquaredScaledError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    0.16914781383660782

    References
    ----------
    M5 Competition Guidelines.
    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average", sp=1, square_root=False):
        name = "MedianSquaredScaledError"
        func = median_squared_scaled_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            sp=sp,
            square_root=square_root,
        )


class MeanAbsoluteError(_BaseForecastingErrorMetric):
    """Mean absolute error (MAE).

    MAE output is non-negative floating point. The best value is 0.0.

    MAE is on the same scale as the data. Because MAE takes the absolute value
    of the forecast error rather than squaring it, MAE penalizes large errors
    to a lesser degree than MSE or RMSE.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MedianAbsoluteError
    MeanSquaredError
    MedianSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mae = MeanAbsoluteError()
    >>> mae(y_true, y_pred)
    0.55
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mae(y_true, y_pred)
    0.75
    >>> mae = MeanAbsoluteError(multioutput='raw_values')
    >>> mae(y_true, y_pred)
    array([0.5, 1. ])
    >>> mae = MeanAbsoluteError(multioutput=[0.3, 0.7])
    >>> mae(y_true, y_pred)
    0.85

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average"):
        name = "MeanAbsoluteError"
        func = mean_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class MedianAbsoluteError(_BaseForecastingErrorMetric):
    """Median absolute error (MdAE).

    MdAE output is non-negative floating point. The best value is 0.0.

    Like MAE, MdAE is on the same scale as the data. Because MAE takes the
    absolute value of the forecast error rather than squaring it, MAE penalizes
    large errors to a lesser degree than MdSE or RdMSE.

    Taking the median instead of the mean of the absolute errors also makes
    this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsoluteError
    MeanSquaredError
    MedianSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdae = MedianAbsoluteError()
    >>> mdae(y_true, y_pred)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdae(y_true, y_pred)
    0.75
    >>> mdae = MedianAbsoluteError(multioutput='raw_values')
    >>> mdae(y_true, y_pred)
    array([0.5, 1. ])
    >>> mdae = MedianAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdae(y_true, y_pred)
    0.85

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average"):
        name = "MedianAbsoluteError"
        func = median_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class MeanSquaredError(_SquaredForecastingErrorMetric):
    """Mean squared error (MSE) or root mean squared error (RMSE).

    If `square_root` is False then calculates MSE and if `square_root` is True
    then RMSE is calculated.  Both MSE and RMSE are both non-negative floating
    point. The best value is 0.0.

    MSE is measured in squared units of the input data, and RMSE is on the
    same scale as the data. Because MSE and RMSE square the forecast error
    rather than taking the absolute value, they penalize large errors more than
    MAE.

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

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    square_root : bool
        Stores whether to take the square root of the

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsoluteError
    MedianAbsoluteError
    MedianSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mse = MeanSquaredError()
    >>> mse(y_true, y_pred)
    0.4125
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mse(y_true, y_pred)
    0.7083333333333334
    >>> rmse = MeanSquaredError(square_root=True)
    >>> rmse(y_true, y_pred)
    0.8227486121839513
    >>> rmse = MeanSquaredError(multioutput='raw_values')
    >>> rmse(y_true, y_pred)
    array([0.41666667, 1.        ])
    >>> rmse = MeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rmse(y_true, y_pred)
    array([0.64549722, 1.        ])
    >>> rmse = MeanSquaredError(multioutput=[0.3, 0.7])
    >>> rmse(y_true, y_pred)
    0.825
    >>> rmse = MeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmse(y_true, y_pred)
    0.8936491673103708

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average", square_root=False):
        name = "MeanSquaredError"
        func = mean_squared_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            square_root=square_root,
        )


class MedianSquaredError(_SquaredForecastingErrorMetric):
    """Median squared error (MdSE) or root median squared error (RMdSE).

    If `square_root` is False then calculates MdSE and if `square_root` is True
    then RMdSE is calculated. Both MdSE and RMdSE return non-negative floating
    point. The best value is 0.0.

    Like MSE, MdSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE. Because MdSE and RMdSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than MAE or MdAE.

    Taking the median instead of the mean of the squared errors makes
    this metric more robust to error outliers relative to a meean based metric
    since the median tends to be a more robust measure of central tendency in
    the presence of outliers.

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

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    square_root : bool
        Stores whether to take the square root of the metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsoluteError
    MedianAbsoluteError
    MeanSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdse = MedianSquaredError()
    >>> mdse(y_true, y_pred)
    0.25
    >>> rmdse = MedianSquaredError(square_root=True)
    >>> rmdse(y_true, y_pred)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdse(y_true, y_pred)
    0.625
    >>> rmdse(y_true, y_pred)
    0.75
    >>> mdse = MedianSquaredError(multioutput='raw_values')
    >>> mdse(y_true, y_pred)
    array([0.25, 1.  ])
    >>> rmdse = MedianSquaredError(multioutput='raw_values', square_root=True)
    >>> rmdse(y_true, y_pred)
    array([0.5, 1. ])
    >>> mdse = MedianSquaredError(multioutput=[0.3, 0.7])
    >>> mdse(y_true, y_pred)
    0.7749999999999999
    >>> rmdse = MedianSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmdse(y_true, y_pred)
    0.85

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average", square_root=False):
        name = "MedianSquaredError"
        func = median_squared_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            square_root=square_root,
        )


class MeanAbsolutePercentageError(_PercentageForecastingErrorMetric):
    """Mean absolute percentage error (MAPE) or symmetric version.

    If `symmetric` is False then calculates MAPE and if `symmetric` is True
    then calculates symmetric mean absolute percentage error (sMAPE). Both
    MAPE and sMAPE output is non-negative floating point. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = True
        Whether to calculate the symmetric version of the percentage metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    symmetric : bool
        Stores whether to calculate the symmetric version of the percentage metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MedianAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MeanAbsolutePercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mape = MeanAbsolutePercentageError(symmetric=False)
    >>> mape(y_true, y_pred)
    0.33690476190476193
    >>> smape = MeanAbsolutePercentageError()
    >>> smape(y_true, y_pred)
    0.5553379953379953
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mape(y_true, y_pred)
    0.5515873015873016
    >>> smape(y_true, y_pred)
    0.6080808080808081
    >>> mape = MeanAbsolutePercentageError(multioutput='raw_values', symmetric=False)
    >>> mape(y_true, y_pred)
    array([0.38095238, 0.72222222])
    >>> smape = MeanAbsolutePercentageError(multioutput='raw_values')
    >>> smape(y_true, y_pred)
    array([0.71111111, 0.50505051])
    >>> mape = MeanAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mape(y_true, y_pred)
    0.6198412698412699
    >>> smape = MeanAbsolutePercentageError(multioutput=[0.3, 0.7])
    >>> smape(y_true, y_pred)
    0.5668686868686869

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average", symmetric=True):
        name = "MeanAbsolutePercentageError"
        func = mean_absolute_percentage_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            symmetric=symmetric,
        )


class MedianAbsolutePercentageError(_PercentageForecastingErrorMetric):
    """Median absolute percentage error (MdAPE) or symmetric version.

    If `symmetric` is False then calculates MdAPE and if `symmetric` is True
    then calculates symmetric median absolute percentage error (sMdAPE). Both
    MdAPE and sMdAPE output is non-negative floating point. The best value is 0.0.

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because it takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = True
        Whether to calculate the symmetric version of the percentage metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    symmetric : bool
        Stores whether to calculate the symmetric version of the percentage metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MedianAbsolutePercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdape = MedianAbsolutePercentageError(symmetric=False)
    >>> mdape(y_true, y_pred)
    0.16666666666666666
    >>> smdape = MedianAbsolutePercentageError()
    >>> smdape(y_true, y_pred)
    0.18181818181818182
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdape(y_true, y_pred)
    0.5714285714285714
    >>> smdape(y_true, y_pred)
    0.39999999999999997
    >>> mdape = MedianAbsolutePercentageError(multioutput='raw_values', symmetric=False)
    >>> mdape(y_true, y_pred)
    array([0.14285714, 1.        ])
    >>> smdape = MedianAbsolutePercentageError(multioutput='raw_values')
    >>> smdape(y_true, y_pred)
    array([0.13333333, 0.66666667])
    >>> mdape = MedianAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mdape(y_true, y_pred)
    0.7428571428571428
    >>> smdape = MedianAbsolutePercentageError(multioutput=[0.3, 0.7])
    >>> smdape(y_true, y_pred)
    0.5066666666666666

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average", symmetric=True):
        name = "MedianAbsolutePercentageError"
        func = median_absolute_percentage_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            symmetric=symmetric,
        )


class MeanSquaredPercentageError(_SquaredPercentageForecastingErrorMetric):
    """Mean squared percentage error (MSPE)  or square root version.

    If `square_root` is False then calculates MSPE and if `square_root` is True
    then calculates root mean squared percentage error (RMSPE). If `symmetric`
    is True then calculates sMSPE or sRMSPE. Output is non-negative floating
    point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = True
        Whether to calculate the symmetric version of the percentage metric

    square_root : bool, default = False
        Whether to take the square root of the metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    symmetric : bool
        Stores whether to calculate the symmetric version of the percentage metric

    square_root : bool
        Stores whether to take the square root of the metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MedianSquaredPercentageError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MeanSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mspe = MeanSquaredPercentageError(symmetric=False)
    >>> mspe(y_true, y_pred)
    0.23776218820861678
    >>> smspe = MeanSquaredPercentageError(square_root=True, symmetric=False)
    >>> smspe(y_true, y_pred)
    0.48760864246710883
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mspe(y_true, y_pred)
    0.5080309901738473
    >>> smspe(y_true, y_pred)
    0.7026794936195895
    >>> mspe = MeanSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mspe(y_true, y_pred)
    array([0.34013605, 0.67592593])
    >>> smspe = MeanSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    array([0.58321184, 0.82214714])
    >>> mspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mspe(y_true, y_pred)
    0.5751889644746787
    >>> smspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    0.7504665536595034

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(
        self, multioutput="uniform_average", symmetric=True, square_root=False
    ):
        name = "MeanSquaredPercentageError"
        func = mean_squared_percentage_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            symmetric=symmetric,
            square_root=square_root,
        )


class MedianSquaredPercentageError(_SquaredPercentageForecastingErrorMetric):
    """Median squared percentage error (MdSPE)  or square root version.

    If `square_root` is False then calculates MdSPE and if `square_root` is True
    then calculates root median squared percentage error (RMdSPE). If `symmetric`
    is True then calculates sMdSPE or sRMdSPE. Output is non-negative floating
    point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    RMdSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = True
        Whether to calculate the symmetric version of the percentage metric

    square_root : bool, default = False
        Whether to take the square root of the metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    symmetric : bool
        Stores whether to calculate the symmetric version of the percentage metric

    square_root : bool
        Stores whether to take the square root of the metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MeanSquaredPercentageError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MedianSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdspe = MedianSquaredPercentageError(symmetric=False)
    >>> mdspe(y_true, y_pred)
    0.027777777777777776
    >>> smdspe = MedianSquaredPercentageError(square_root=True, symmetric=False)
    >>> smdspe(y_true, y_pred)
    0.16666666666666666
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdspe(y_true, y_pred)
    0.5102040816326531
    >>> smdspe(y_true, y_pred)
    0.5714285714285714
    >>> mdspe = MedianSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mdspe(y_true, y_pred)
    array([0.02040816, 1.        ])
    >>> smdspe = MedianSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smdspe(y_true, y_pred)
    array([0.14285714, 1.        ])
    >>> mdspe = MedianSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mdspe(y_true, y_pred)
    0.7061224489795918
    >>> smdspe = MedianSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smdspe(y_true, y_pred)
    0.7428571428571428

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(
        self, multioutput="uniform_average", symmetric=True, square_root=False
    ):
        name = "MedianSquaredPercentageError"
        func = median_squared_percentage_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            symmetric=symmetric,
            square_root=square_root,
        )


class MeanRelativeAbsoluteError(_BaseForecastingErrorMetric):
    """Mean relative absolute error (MRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MRAE applies mean absolute error (MAE) to the resulting relative errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae = MeanRelativeAbsoluteError()
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.9511111111111111
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8703703703703702
    >>> mrae = MeanRelativeAbsoluteError(multioutput='raw_values')
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.51851852, 1.22222222])
    >>> mrae = MeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    1.0111111111111108

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "MeanRelativeAbsoluteError"
        func = mean_relative_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class MedianRelativeAbsoluteError(_BaseForecastingErrorMetric):
    """Median relative absolute error (MdRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MdRAE applies medan absolute error (MdAE) to the resulting relative errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae = MedianRelativeAbsoluteError()
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    1.0
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.6944444444444443
    >>> mdrae = MedianRelativeAbsoluteError(multioutput='raw_values')
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.55555556, 0.83333333])
    >>> mdrae = MedianRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.7499999999999999

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "MedianRelativeAbsoluteError"
        func = median_relative_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class GeometricMeanRelativeAbsoluteError(_BaseForecastingErrorMetric):
    """Geometric mean relative absolute error (GMRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRAE applies geometric mean absolute error (GMAE) to the resulting relative
    errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae = GeometricMeanRelativeAbsoluteError()
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.0007839273064064755
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.5578632807409556
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput='raw_values')
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.97801163e-06, 1.11572158e+00])
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.7810066018326863

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "GeometricMeanRelativeAbsoluteError"
        func = geometric_mean_relative_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class GeometricMeanRelativeSquaredError(_SquaredForecastingErrorMetric):
    """Geometric mean relative squared error (GMRSE).

    If `square_root` is False then calculates GMRSE and if `square_root` is True
    then calculates root geometric mean relative squared error (RGMRSE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRSE applies geometric mean squared error (GMSE) to the resulting relative
    errors. RGMRSE applies root geometric mean squared error (RGMSE) to the
    resulting relative errors.

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

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    square_root : bool
        Stores whether to take the square root of the metric

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    See Also
    --------
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse = GeometricMeanRelativeSquaredError()
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.0008303544925949156
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.622419372049448
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput='raw_values')
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.09227746e-06, 1.24483465e+00])
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput=[0.3, 0.7])
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8713854839582426

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average", square_root=False):
        name = "GeometricMeanRelativeSquaredError"
        func = geometric_mean_relative_squared_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            square_root=square_root,
        )


class MeanAsymmetricError(_AsymmetricForecastingErrorMetric):
    """Calculate asymmetric loss function.

    Error values that are less than the asymmetric threshold have
    `left_error_function` applied. Error values greater than or equal to
    asymmetric threshold  have `right_error_function` applied.

    Many forecasting loss functions assume that over- and under-
    predictions should receive an equal penalty. However, this may not align
    with the actual cost faced by users' of the forecasts. Asymmetric loss
    functions are useful when the cost of under- and over- prediction are not
    the same.

    Setting `asymmetric_threshold` to zero, `left_error_function` to 'squared'
    and `right_error_function` to 'absolute` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for `left_error_function` set to 'absolute' and `right_error_function`
    set to 'squared`.

    Parameters
    ----------
    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have `left_error_function`
        applied. Error values greater than or equal to asymmetric threshold
        have `right_error_function` applied.

    left_error_function : str, {'squared', 'absolute'}
        Loss penalty to apply to error values less than the asymmetric threshold.

    right_error_function : str, {'squared', 'absolute'}
        Loss penalty to apply to error values greater than or equal to the
        asymmetric threshold.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    asymmetric_threshold : numeric
        Stores threshold to use applying asymmetric loss to errors

    left_error_function : str
        Stores loss penalty to apply to error values less than the asymmetric threshold.

    right_error_function : str
        Stores loss penalty to apply to error values greater than or equal to
        the asymmetric threshold.

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAsymmetricError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> masymmetric = MeanAsymmetricError()
    >>> masymmetric(y_true, y_pred)
    0.5
    >>> masymmetric = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> masymmetric(y_true, y_pred)
    0.4625
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> masymmetric = MeanAsymmetricError()
    >>> masymmetric(y_true, y_pred)
    0.75
    >>> masymmetric = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> masymmetric(y_true, y_pred)
    0.7083333333333334
    >>> masymmetric = MeanAsymmetricError(multioutput='raw_values')
    >>> masymmetric(y_true, y_pred)
    array([0.5, 1. ])
    >>> masymmetric = MeanAsymmetricError(multioutput=[0.3, 0.7])
    >>> masymmetric(y_true, y_pred)
    0.85

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
    Thomson, South-Western: Ohio, US.
    """

    def __init__(
        self,
        multioutput="uniform_average",
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
    ):
        name = "MeanAsymmetricError"
        func = mean_asymmetric_error
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            asymmetric_threshold=asymmetric_threshold,
            left_error_function=left_error_function,
            right_error_function=right_error_function,
        )


class RelativeLoss(_RelativeLossForecastingErrorMetric):
    """Calculate relative loss of forecast versus benchmark forecast.

    Applies a forecasting performance metric to a set of forecasts and
    benchmark forecasts and reports ratio of the metric from the forecasts to
    the the metric from the benchmark forecasts. Relative loss output is
    non-negative floating point. The best value is 0.0.

    If the score of the benchmark predictions for a given loss function is zero
    then a large value is returned.

    This function allows the calculation of scale-free relative loss metrics.
    Unlike mean absolute scaled error (MASE) the function calculates the
    scale-free metric relative to a defined loss function on a benchmark
    method instead of the in-sample training data. Like MASE, metrics created
    using this function can be used to compare forecast methods on a single
    series and also to compare forecast accuracy between series.

    This is useful when a scale-free comparison is beneficial but the training
    data used to generate some (or all) predictions is unknown such as when
    comparing the loss of 3rd party forecasts or surveys of professional
    forecasters.

    Only metrics that do not require y_train are curretnly supported.

    Parameters
    ----------
    relative_loss_function : function
        Function to use in calculation relative loss.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Attributes
    ----------
    name : str
        The name of the loss metric

    greater_is_better : bool
        Stores whether the metric is optimized by minimization or maximization.
        If False, minimizing the metric is optimal.
        If True, maximizing the metric is optimal.

    relative_loss_function : function
        Stores function used to calculate relative loss

    multioutput : str
        Stores how the metric should aggregate multioutput data.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import RelativeLoss
    >>> from sktime.performance_metrics.forecasting import mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8148148148148147
    >>> relative_mse = RelativeLoss(relative_loss_function=mean_squared_error)
    >>> relative_mse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.5178095088655261
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8490566037735847
    >>> relative_mae = RelativeLoss(multioutput='raw_values')
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.625     , 1.03448276])
    >>> relative_mae = RelativeLoss(multioutput=[0.3, 0.7])
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.927272727272727

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(
        self, multioutput="uniform_average", relative_loss_function=mean_absolute_error
    ):
        name = "RelativeLoss"
        func = relative_loss
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput,
            relative_loss_function=relative_loss_function,
        )
