# -*- coding: utf-8 -*-
from sktime.performance_metrics.forecasting._functions import (
    relative_loss,
    asymmetric_error,
    mean_absolute_scaled_error,
    median_absolute_scaled_error,
    root_mean_squared_scaled_error,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    median_absolute_error,
    median_squared_error,
    root_median_squared_error,
    symmetric_mean_absolute_percentage_error,
    symmetric_median_absolute_percentage_error,
    mean_absolute_percentage_error,
    median_absolute_percentage_error,
    mean_squared_percentage_error,
    median_squared_percentage_error,
    root_mean_squared_percentage_error,
    root_median_squared_percentage_error,
    mean_relative_absolute_error,
    median_relative_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
)

__author__ = ["Markus Löning", "Tomasz Chodakowski", "Ryan Kuhns"]
__all__ = [
    "MetricFunctionWrapper",
    "make_forecasting_scorer",
    "RelativeLoss",
    "AsymmetricError",
    "MeanAbsoluteScaledError",
    "MedianAbsoluteScaledError",
    "RootMeanSquaredScaledError",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "RootMeanSquaredError",
    "MedianAbsoluteError",
    "MedianSquaredError",
    "RootMedianSquaredError",
    "SymmetricMeanAbsolutePercentageError",
    "SymmetricMedianAbsolutePercentageError",
    "MeanAbsolutePercentageError",
    "MedianAbsolutePercentageError",
    "MeanSquaredPercentageError",
    "MedianSquaredPercentageError",
    "RootMeanSquaredPercentageError",
    "RootMedianSquaredPercentageError",
    "MeanRelativeAbsoluteError",
    "MedianRelativeAbsoluteError",
    "GeometricMeanRelativeAbsoluteError",
    "GeometricMeanRelativeSquaredError",
]


class MetricFunctionWrapper:
    def __init__(self, fn, name=None, greater_is_better=False):
        self.fn = fn
        self.name = name if name is not None else fn.__name__
        self.greater_is_better = greater_is_better

    def __call__(self, y_test, y_pred, *args, **kwargs):
        return self.fn(y_test, y_pred, *args, **kwargs)


def make_forecasting_scorer(fn, name=None, greater_is_better=False):
    """Factory method for creating metric classes from metric functions"""
    return MetricFunctionWrapper(fn, name=name, greater_is_better=greater_is_better)


class RelativeLoss(MetricFunctionWrapper):
    def __init__(self):
        name = "RelativeLoss"
        fn = relative_loss
        greater_is_better = False
        super(RelativeLoss, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class AsymmetricError(MetricFunctionWrapper):
    def __init__(self):
        name = "AsymmetricError"
        fn = asymmetric_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MeanAbsoluteScaledError(MetricFunctionWrapper):
    def __init__(self):
        name = "MeanAbsoluteScaledError"
        fn = mean_absolute_scaled_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MedianAbsoluteScaledError(MetricFunctionWrapper):
    def __init__(self):
        name = "MedianAbsoluteScaledError"
        fn = median_absolute_scaled_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class RootMeanSquaredScaledError(MetricFunctionWrapper):
    def __init__(self):
        name = "RootMeanSquaredScaledError"
        fn = root_mean_squared_scaled_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MeanAbsoluteError(MetricFunctionWrapper):
    def __init__(self):
        name = "MeanAbsoluteError"
        fn = mean_absolute_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MeanSquaredError(MetricFunctionWrapper):
    def __init__(self):
        name = "MeanSquaredError"
        fn = mean_squared_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class RootMeanSquaredError(MetricFunctionWrapper):
    def __init__(self):
        name = "RootMeanSquaredError"
        fn = root_mean_squared_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MedianAbsoluteError(MetricFunctionWrapper):
    def __init__(self):
        name = "MedianAbsoluteError"
        fn = median_absolute_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MedianSquaredError(MetricFunctionWrapper):
    def __init__(self):
        name = "MedianSquaredError"
        fn = median_squared_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class RootMedianSquaredError(MetricFunctionWrapper):
    def __init__(self):
        name = "RootMedianSquaredError"
        fn = root_median_squared_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class SymmetricMeanAbsolutePercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "SymmetricMeanAbsolutePercentageError"
        fn = symmetric_mean_absolute_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class SymmetricMedianAbsolutePercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "SymmetricMedianAbsolutePercentageError"
        fn = symmetric_median_absolute_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MeanAbsolutePercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "MeanAbsolutePercentageError"
        fn = mean_absolute_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MedianAbsolutePercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "MedianAbsolutePercentageError"
        fn = median_absolute_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MeanSquaredPercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "MeanSquaredPercentageError"
        fn = mean_squared_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MedianSquaredPercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "MedianSquaredPercentageError"
        fn = median_squared_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class RootMeanSquaredPercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "RootMeanSquaredPercentageError"
        fn = root_mean_squared_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class RootMedianSquaredPercentageError(MetricFunctionWrapper):
    def __init__(self):
        name = "RootMedianSquaredPercentageError"
        fn = root_median_squared_percentage_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MeanRelativeAbsoluteError(MetricFunctionWrapper):
    def __init__(self):
        name = "MeanRelativeAbsoluteError"
        fn = mean_relative_absolute_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MedianRelativeAbsoluteError(MetricFunctionWrapper):
    def __init__(self):
        name = "MedianRelativeAbsoluteError"
        fn = median_relative_absolute_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class GeometricMeanRelativeAbsoluteError(MetricFunctionWrapper):
    def __init__(self):
        name = "GeometricMeanRelativeAbsoluteError"
        fn = geometric_mean_relative_absolute_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class GeometricMeanRelativeSquaredError(MetricFunctionWrapper):
    def __init__(self):
        name = "GeometricMeanRelativeSquaredError"
        fn = geometric_mean_relative_squared_error
        greater_is_better = False
        super(AsymmetricError, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )
