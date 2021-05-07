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
    """Base class for defining forecasting error metrics in sktime. Extends
    sktime's BaseMetric to the forecasting interface.
    """

    _tags = {
        "requires_y_train": False,
        "requires_y_pred_benchmark": False,
        "univariate-only": False,
    }

    greater_is_better = False

    def __init__(self, func, name=None, multioutput="uniform_average"):
        self.multioutput = multioutput
        super().__init__(func, name=name)

    def __call__(self, y_true, y_pred, **kwargs):
        """Returns calculated loss metric by passing `y_true` and `y_pred` to
        underlying metric function.

        Parameters
        ----------
        y_true : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Estimated target values.

        y_train : pandas Series
            Optional keyword argument to pass training data.

        y_pred_benchmark : pandas Series
            Optional keyword argument to pass benchmark predictions

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self._func(y_true, y_pred, multioutput=self.multioutput, **kwargs)


class _BaseForecastingSuccessMetric(_BaseForecastingErrorMetric):
    greater_is_better = True


class _PercentageErrorMixin:
    def __call__(self, y_true, y_pred, **kwargs):
        """Returns calculated loss metric by passing `y_true` and `y_pred` to
        underlying metric function.

        Uses `symmetric` attribute to determine whether underlying function
        should return symmetric percentage error metric or a percentage error
        metric.

        Parameters
        ----------
        y_true : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Estimated target values.

        y_train : pandas Series, default=None
            Optional keyword argument to pass training data.

        y_pred_benchmark : pandas Series
            Optional keyword argument to pass benchmark predictions

        Returns
        -------
        loss : float
            Calculated loss metric
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
        """Returns calculated loss metric by passing `y_true` and `y_pred` to
        underlying metric function.

        Uses `square_root` attribute to determine whether the
        underlying function should return the square_root of the metric or
        the metric.

        Parameters
        ----------
        y_true : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Estimated target values.

        y_train : pandas Series, default=None
            Optional keyword argument to pass training data.

        y_pred_benchmark : pandas Series
            Optional keyword argument to pass benchmark predictions

        Returns
        -------
        loss : float
            Calculated loss metric
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
        """Returns calculated loss metric by passing `y_true` and `y_pred` to
        underlying metric function.

        Uses `symmetric` attribute to determine whether underlying function
        should return symmetric percentage error metric or a percentage error
        metric. Also uses `square_root` attribute to determine whether the
        underlying function should return the square_root of the metric or
        the metric.

        Parameters
        ----------
        y_true : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Estimated target values.

        y_train : pandas Series
            Optional keyword argument to pass training data.

        y_pred_benchmark : pandas Series
            Optional keyword argument to pass benchmark predictions

        Returns
        -------
        loss : float
            Calculated loss metric.
            If square_root = True, tells .
            I
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
        """Returns calculated loss metric by passing `y_true` and `y_pred` to
        underlying metric function.

        Parameters
        ----------
        y_true : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Estimated target values.

        y_train : pandas Series
            Optional keyword argument to pass training data.


        Returns
        -------
        loss : float
            Calculated loss metric
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
        """Returns calculated loss metric by passing `y_true` and `y_pred` to
        underlying metric function.

        Parameters
        ----------
        y_true : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
                where fh is the forecasting horizon
            Estimated target values.

        y_pred_benchmark : pandas Series
            Optional keyword argument to pass benchmark predictions

        Returns
        -------
        loss : float
            Calculated loss metric
        """
        return self._func(
            y_true,
            y_pred,
            multioutput=self.multioutput,
            loss_function=self.relative_loss_function,
            **kwargs,
        )


class _ScaledForecastingErrorMetric(_BaseForecastingErrorMetric):
    _tags = {
        "requires_y_train": True,
        "requires_y_pred_benchmark": False,
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
        "requires_y_train": False,
        "requires_y_pred_benchmark": True,
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
    """Factory method for creating metric classes from metric functions

    Parameters
    ----------
    func:
        Loss function to convert to a forecasting scorer class

    name: str, default=None
        Name to use for the forecasting scorer loss class

    greater_is_better: bool, default=False
        If True then maximizing the metric is better.
        If False then minimizing the metric is better.

    Returns
    -------
    scorer:
        Metric class that can be used as forecasting scorer.
    """
    if greater_is_better:
        return _BaseForecastingErrorMetric(func, name=name, multioutput=multioutput)
    else:
        return _BaseForecastingSuccessMetric(func, name=name, multioutput=multioutput)


class MeanAbsoluteScaledError(_ScaledForecastingErrorMetric):
    """Mean absolute scaled error (MASE). MASE output is non-negative floating
    point. The best value is 0.0.

    This scale-free error metric can be used to compare forecast methods on
    a single series and also to compare forecast accuracy between series.

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    ..[2] Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
          for intermittent demand", Foresight, Issue 4.
    ..[3] Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
          "The M4 Competition: 100,000 time series and 61 forecasting methods",
          International Journal of Forecasting, Volume 3
    """

    def __init__(self, multioutput="uniform_average", sp=1):
        name = "MeanAbsoluteScaledError"
        func = mean_absolute_scaled_error
        super().__init__(func=func, name=name, multioutput=multioutput, sp=sp)


class MedianAbsoluteScaledError(_ScaledForecastingErrorMetric):
    """Median absolute scaled error (MdASE). MdASE output is non-negative
    floating point. The best value is 0.0.

    Taking the median instead of the mean of the test and train absolute errors
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Like MASE, this scale-free error metric can be used to compare forecast
    methods on a single series and also to compare forecast accuracy between
    series.

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    ..[2] Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
          for intermittent demand", Foresight, Issue 4.
    ..[3] Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
          "The M4 Competition: 100,000 time series and 61 forecasting methods",
          International Journal of Forecasting, Volume 3
    """

    def __init__(self, multioutput="uniform_average", sp=1):
        name = "MedianAbsoluteScaledError"
        func = median_absolute_scaled_error
        super().__init__(func=func, name=name, multioutput=multioutput, sp=sp)


class MeanSquaredScaledError(_ScaledSquaredForecastingErrorMetric):
    """Mean squared scaled error (MSSE) `square_root` is False or
    root mean squared scaled error (RMSSE) if `square_root` is True.
    MSSE and RMSSE output is non-negative floating point. The best value is 0.0.

    This is a squared varient of the MASE loss metric. Like MASE this
    scale-free metric can be used to copmare forecast methods on a single
    series or between series.

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] M5 Competition Guidelines.
          https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx
    ..[2] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Median squared scaled error (MdSSE) if `square_root` is False or
    root median squared scaled error (RMdSSE) if `square_root` is True.
    MdSSE and RMdSSE output is non-negative floating point. The best value is 0.0.

    This is a squared varient of the MdASE loss metric. Like MASE, MdASE, MSSE
    and RMSSE this scale-free metric can be used to compare forecast methods on a
    single series or between series.

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] M5 Competition Guidelines.
          https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx
    ..[2] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Mean absolute error (MAE). MAE output is non-negative floating point.
    The best value is 0.0.

    MAE is on the same scale as the data. Because it takes the absolute value
    of the forecast error rather than the square, it is less sensitive to
    outliers than MSE or RMSE.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average"):
        name = "MeanAbsoluteError"
        func = mean_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class MedianAbsoluteError(_BaseForecastingErrorMetric):
    """Median absolute error (MdAE).  MdAE output is non-negative floating
    point. The best value is 0.0.

    Like MAE, MdAE is on the same scale as the data. Because it takes the
    absolute value of the forecast error rather than the square, it is less
    sensitive to outliers than MSE, MdSE, RMSE or RMdSE.

    Taking the median instead of the mean of the absolute errors also makes
    this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    """

    def __init__(self, multioutput="uniform_average"):
        name = "MedianAbsoluteError"
        func = median_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class MeanSquaredError(_SquaredForecastingErrorMetric):
    """Mean squared error (MSE) if `square_root` is False or
    root mean squared error (RMSE)  if `square_root` if True. MSE and RMSE are
    both non-negative floating point. The best value is 0.0.

    MSE is measured in squared units of the input data, and RMSE is on the
    same scale as the data. Because both metrics squares the
    forecast error rather than taking the absolute value, they are more sensitive
    to outliers than MAE or MdAE.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Median squared error (MdSE) if `square_root` is False or root median
    squared error (RMdSE) if `square_root` is True. MdSE and RMdSE return
    non-negative floating point. The best value is 0.0.

    Like MSE, MdSE is measured in squared units of the input data. RMdSe is
    on the same scale as the input data like RMSE. Because MdSE and RMdSE
    square the forecast error rather than taking the absolute value, they are
    more sensitive to outliers than MAE or MdAE.

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Mean absolute percentage error (MAPE) if `symmetric` is False or
    symmetric mean absolute percentage error (sMAPE) if `symmetric is True.
    MAPE and sMAPE output is non-negative floating point. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it is less sensitive to outliers than MSPE, RMSPE, MdSPE or RMdSPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = True
        Whether to calculate the symmetric version of the percentage metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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
    MedianAbsoulutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Median absolute percentage error (MdAPE) if `symmetric` is False or
    symmetric median absolute percentage error (sMdAPE). MdAPE and sMdAPE output
    is non-negative floating point. The best value is 0.0.

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because it takes the absolute value rather than square the percentage forecast
    error, it is less sensitive to outliers than MSPE, RMSPE, MdSPE or RMdSPE.

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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
    MeanAbsoulutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Mean squared percentage error (MSPE) if `square_root` is False or
    root mean squared percentage error (RMSPE) if `square_root` is True.
    MSPE and RMSPE output is non-negative floating point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because either calculation takes the square rather than absolute value of
    the percentage forecast error, they are more sensitive to outliers than
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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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
    MeanAbsoulutePercentageError
    MedianAbsolutePercentageError
    MedianSquaredPercentageError

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
    """Median squared percentage error (MdSPE) if `square_root` is False or
    root median squared percentage error (RMdSPE) if `square_root` is True.
    MdSPE and RMdSPE output is non-negative floating point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    RMdSPE is measured in percentage error relative to the test data.
    Because it takes the square rather than absolute value of the percentage
    forecast error, both calculations are more sensitive to outliers than
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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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
    MeanAbsoulutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires_y_train": False,
        "requires_y_pred_benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "MeanRelativeAbsoluteError"
        func = mean_relative_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class MedianRelativeAbsoluteError(_BaseForecastingErrorMetric):
    """Median relative absolute error (MdRAE).

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires_y_train": False,
        "requires_y_pred_benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "MedianRelativeAbsoluteError"
        func = median_relative_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class GeometricMeanRelativeAbsoluteError(_BaseForecastingErrorMetric):
    """Geometric mean relative absolute error (GMRAE).

    Parameters
    ---------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires_y_train": False,
        "requires_y_pred_benchmark": True,
        "univariate-only": False,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "GeometricMeanRelativeAbsoluteError"
        func = geometric_mean_relative_absolute_error
        super().__init__(func=func, name=name, multioutput=multioutput)


class GeometricMeanRelativeSquaredError(_SquaredForecastingErrorMetric):
    """Geometric mean relative squared error (GMRSE) if `square_root` is False or
    root geometric mean relative squared error (RGMRSE) if `square_root` is True.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
    """

    _tags = {
        "requires_y_train": False,
        "requires_y_pred_benchmark": True,
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
    """Calculates asymmetric loss function. Error values that are less
    than the asymmetric threshold have `left_error_function` applied.
    Error values greater than or equal to asymmetric threshold  have
    `right_error_function` applied.

    Many forecasting loss functions assume that over- and under-
    predictions should receive an equal penalty. However, this may not align
    with the actual cost faced by users' of the forecasts. Asymmetric loss
    functions are useful when the cost of under- and over- prediction are not
    the same.

    Setting `asymmetric_threshold` to zero, `left_error_function` to 'squared'
    and `right_error_function` to 'absoulte` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for `left_error_function` set to 'absolute' and `right_error_function`
    set to 'squared`

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
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    ..[2] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)" ,
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
    """Calculates relative loss for a set of predictions and benchmark
    predictions for a given loss function.

    This function allows the calculation of scale-free relative loss metrics.
    Unlike mean absolute scaled error (MASE) the function calculates the
    scale-free metric relative to a defined loss function on a benchmark
    method. Like MASE, metrics created using this function can be used to compare
    forecast methods on a single series and also to compare forecast accuracy
    between series.

    This is useful when a scale-free comparison is beneficial but the training
    used to generate some (or all) predictions is unknown such as when
    comparing the loss of 3rd party forecasts or surveys of professional
    forecastsers.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    relative_loss_function : function
        Function to use in calculation relative loss.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Returns
    -------
    relative_loss : float
        Loss for a method relative to loss for a benchmark method for a given
        loss metric.

    References
    ----------
    ..[1] Hyndman, R. J and Koehler, A. B. (2006).
          "Another look at measures of forecast accuracy", International
          Journal of Forecasting, Volume 22, Issue 4.
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
