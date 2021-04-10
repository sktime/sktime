# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sktime.performance_metrics.forecasting._functions import (
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
    "PercentageErrorMixIn",
    "SquaredErrorMixIn",
    "SquaredPercentageErrorMixIn",
    "BaseMetricFunctionWrapper",
    "PercentageMetricFunctionWrapper",
    "SquaredMetricFunctionWrapper",
    "SquaredPercentageMetricFunctionWrapper",
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
]


class BaseMetricFunctionWrapper(BaseEstimator):
    def __init__(self, fn, name=None, greater_is_better=False):
        self.fn = fn
        self.name = name if name is not None else fn.__name__
        self.greater_is_better = greater_is_better

    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred)


class PercentageErrorMixIn:
    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred, symmetric=self.symmetric)


class SquaredErrorMixIn:
    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred, square_root=self.square_root)


class SquaredPercentageErrorMixIn:
    def __call__(self, y_true, y_pred):
        return self.fn(
            y_true, y_pred, symmetric=self.symmetric, square_root=self.square_root
        )


class AsymmetricErrorMixIn:
    def __call__(self, y_true, y_pred):
        return self.fn(
            y_true,
            y_pred,
            asymmetric_threshold=self.asymmetric_treshold,
            left_error_function=self.left_error_function,
            right_error_function=self.right_error_function,
        )


class PercentageMetricFunctionWrapper(PercentageErrorMixIn, BaseMetricFunctionWrapper):
    def __init__(self, fn, name=None, greater_is_better=False, symmetric=False):
        self.symmetric = symmetric
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class SquaredMetricFunctionWrapper(SquaredErrorMixIn, BaseMetricFunctionWrapper):
    def __init__(self, fn, name=None, greater_is_better=False, square_root=False):
        self.square_root = square_root
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class SquaredPercentageMetricFunctionWrapper(
    SquaredPercentageErrorMixIn, BaseMetricFunctionWrapper
):
    def __init__(
        self, fn, name=None, greater_is_better=False, square_root=False, symmetric=False
    ):
        self.square_root = square_root
        self.symmetric = symmetric
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class AsymmetricMetricFunctionWrapper(AsymmetricErrorMixIn, BaseMetricFunctionWrapper):
    def __init__(
        self,
        fn,
        name=None,
        greater_is_better=False,
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
    ):
        self.asymmetric_threshold = asymmetric_threshold
        self.left_error_function = left_error_function
        self.right_error_function = right_error_function
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


def make_forecasting_scorer(
    fn, name=None, greater_is_better=False, symmetric=None, square_root=None
):
    """Factory method for creating metric classes from metric functions

    Parameters
    ----------
    fn:
        Loss function to convert to a forecasting scorer class

    name: str, default=None
        Name to use for the forecasting scorer loss class

    greater_is_better: bool, default=False
        If True then maximizing the metric is better.
        If False then minimizing the metric is better.

    symmetric: bool, default=None
        Whether to calculate symmetric percnetage error.
            If None then created metric class does not include a `symmetric`
                parameter
            If True, created metric class includes has `symmetric` attribute
                equal to True. Metric calculates symmetric version of
                percentage error loss function.
            If False, created metric class includes has `symmetric` attribute
                equal to False. Metric calculates standard version of
                percentage error loss function

    square_root: bool, default=None
        Whether to take the square root of the calculated metric.
            If None then created metric class does not include a `square_root`
                parameter
            If True, created metric class includes has `square_root` attribute
                equal to True. Metric calculates square root of provided loss function.
            If False, created metric class includes has `square_root` attribute
                equal to False. Metric calculates provided loss function.

    Returns
    -------
    scorer:
        Metric class that can be used as forecasting scorer.
    """
    # Create bas
    if not symmetric and not square_root:
        return BaseMetricFunctionWrapper(
            fn, name=name, greater_is_better=greater_is_better
        )
    elif symmetric and not square_root:
        return PercentageMetricFunctionWrapper(
            fn, name=name, greater_is_better=greater_is_better, symmetric=symmetric
        )
    elif not symmetric and square_root:
        return SquaredMetricFunctionWrapper(
            fn, name=name, greater_is_better=greater_is_better, square_root=square_root
        )

    elif symmetric and square_root:
        return SquaredPercentageMetricFunctionWrapper(
            fn,
            name=name,
            greater_is_better=greater_is_better,
            symmetric=symmetric,
            square_root=square_root,
        )


class MeanAbsoluteScaledError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "MeanAbsoluteScaledError"
        fn = mean_absolute_scaled_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class MedianAbsoluteScaledError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "MedianAbsoluteScaledError"
        fn = median_absolute_scaled_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class MeanSquaredScaledError(SquaredMetricFunctionWrapper):
    def __init__(self, square_root=False):
        name = "MeanSquaredScaledError"
        fn = mean_squared_scaled_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            square_root=square_root,
        )


class MedianSquaredScaledError(SquaredMetricFunctionWrapper):
    def __init__(self, square_root=False):
        name = "MedianSquaredScaledError"
        fn = median_squared_scaled_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            square_root=square_root,
        )


class MeanAbsoluteError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "MeanAbsoluteError"
        fn = mean_absolute_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class MedianAbsoluteError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "MedianAbsoluteError"
        fn = median_absolute_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class MeanSquaredError(SquaredMetricFunctionWrapper):
    def __init__(self, square_root=False):
        name = "MeanSquaredError"
        fn = mean_squared_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            square_root=square_root,
        )


class MedianSquaredError(SquaredMetricFunctionWrapper):
    def __init__(self, square_root=False):
        name = "MedianSquaredError"
        fn = median_squared_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            square_root=square_root,
        )


class MeanAbsolutePercentageError(PercentageMetricFunctionWrapper):
    def __init__(self, symmetric=False):
        name = "MeanAbsolutePercentageError"
        fn = mean_absolute_percentage_error
        greater_is_better = False
        super().__init__(
            fn=fn, name=name, greater_is_better=greater_is_better, symmetric=symmetric
        )


class MedianAbsolutePercentageError(PercentageMetricFunctionWrapper):
    def __init__(self, symmetric=False):
        name = "MedianAbsolutePercentageError"
        fn = median_absolute_percentage_error
        greater_is_better = False
        super().__init__(
            fn=fn, name=name, greater_is_better=greater_is_better, symmetric=symmetric
        )


class MeanSquaredPercentageError(SquaredPercentageMetricFunctionWrapper):
    def __init__(self, symmetric=False, square_root=False):
        name = "MeanSquaredPercentageError"
        fn = mean_squared_percentage_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            symmetric=symmetric,
            square_root=square_root,
        )


class MedianSquaredPercentageError(SquaredPercentageMetricFunctionWrapper):
    def __init__(self, symmetric=False, square_root=False):
        name = "MedianSquaredPercentageError"
        fn = median_squared_percentage_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            symmetric=symmetric,
            square_root=square_root,
        )


class MeanRelativeAbsoluteError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "MeanRelativeAbsoluteError"
        fn = mean_relative_absolute_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class MedianRelativeAbsoluteError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "MedianRelativeAbsoluteError"
        fn = median_relative_absolute_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class GeometricMeanRelativeAbsoluteError(BaseMetricFunctionWrapper):
    def __init__(self):
        name = "GeometricMeanRelativeAbsoluteError"
        fn = geometric_mean_relative_absolute_error
        greater_is_better = False
        super().__init__(fn=fn, name=name, greater_is_better=greater_is_better)


class GeometricMeanRelativeSquaredError(SquaredMetricFunctionWrapper):
    def __init__(self, square_root=False):
        name = "GeometricMeanRelativeSquaredError"
        fn = geometric_mean_relative_squared_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            square_root=square_root,
        )


class MeanAsymmetricError(AsymmetricMetricFunctionWrapper):
    def __init__(
        self,
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
    ):
        name = "MeanAsymmetricError"
        fn = mean_asymmetric_error
        greater_is_better = False
        super().__init__(
            fn=fn,
            name=name,
            greater_is_better=greater_is_better,
            asymmetric_threshold=asymmetric_threshold,
            left_error_function=left_error_function,
            right_error_function=right_error_function,
        )
