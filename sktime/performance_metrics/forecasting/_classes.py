<<<<<<< HEAD
from sktime.performance_metrics.forecasting import mase_loss, smape_loss
=======
from sktime.performance_metrics.forecasting._functions import mase_loss
from sktime.performance_metrics.forecasting._functions import smape_loss
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

__author__ = ['Markus Löning']
__all__ = ["MetricFunctionWrapper", "make_forecasting_scorer", "MASE", "sMAPE"]


class MetricFunctionWrapper:

    def __init__(self, fn, name=None, greater_is_better=False):
        self.fn = fn
        self.name = name if name is not None else fn.__name__
        self.greater_is_better = greater_is_better

    def __call__(self, y_test, y_pred, *args, **kwargs):
        return self.fn(y_test, y_pred, *args, **kwargs)


def make_forecasting_scorer(fn, name=None, greater_is_better=False):
    """Factory method for creating metric classes from metric functions"""
<<<<<<< HEAD
    return MetricFunctionWrapper(fn, name=name, greater_is_better=greater_is_better)
=======
    return MetricFunctionWrapper(fn, name=name,
                                 greater_is_better=greater_is_better)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed


class MASE(MetricFunctionWrapper):

    def __init__(self):
        name = "MASE"
        fn = mase_loss
        greater_is_better = False
<<<<<<< HEAD
        super(MASE, self).__init__(fn=fn, name=name, greater_is_better=greater_is_better)
=======
        super(MASE, self).__init__(fn=fn, name=name,
                                   greater_is_better=greater_is_better)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed


class sMAPE(MetricFunctionWrapper):

    def __init__(self):
        name = "sMAPE"
        fn = smape_loss
        greater_is_better = False
<<<<<<< HEAD
        super(sMAPE, self).__init__(fn=fn, name=name, greater_is_better=greater_is_better)
=======
        super(sMAPE, self).__init__(fn=fn, name=name,
                                    greater_is_better=greater_is_better)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
