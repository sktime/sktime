# -*- coding: utf-8 -*-
from sktime.performance_metrics.forecasting._functions import (
    mase_loss,
    smape_loss,
    mape_loss,
)

__author__ = ["Markus Löning", "Tomasz Chodakowski"]
__all__ = ["MetricFunctionWrapper", "make_forecasting_scorer", "MASE", "sMAPE", "MAPE"]


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


class MASE(MetricFunctionWrapper):
    def __init__(self):
        name = "MASE"
        fn = mase_loss
        greater_is_better = False
        super(MASE, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class sMAPE(MetricFunctionWrapper):
    def __init__(self):
        name = "sMAPE"
        fn = smape_loss
        greater_is_better = False
        super(sMAPE, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )


class MAPE(MetricFunctionWrapper):
    def __init__(self):
        name = "MAPE"
        fn = mape_loss
        greater_is_better = False
        super(sMAPE, self).__init__(
            fn=fn, name=name, greater_is_better=greater_is_better
        )
