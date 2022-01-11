# -*- coding: utf-8 -*-
"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressirs.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_classification", "scenarios_regression"]

from copy import deepcopy

from sktime.base import BaseObject
from sktime.utils._testing.panel import _make_classification_y, _make_panel_X
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


# no logic in scenario classes, but placeholder and for pattern homogeneity
class ClassifierTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

    def get_args(self, key, obj=None, deepcopy_args=False):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        # use same args for predict-like functions as for predict
        if key in ["predict_proba", "decision_function"]:
            key = "predict"

        args = self.args[key]

        if deepcopy_args:
            args = deepcopy(args)

        return args
    pass


y = _make_classification_y(random_state=RAND_SEED)
X = _make_panel_X(n_instances=10, n_timepoints=10, random_state=RAND_SEED, y=y)
X_test = _make_panel_X(n_instances=5, n_timepoints=10, random_state=RAND_SEED)


class ClassifierFitPredict(ClassifierTestScenario):
    """Fit/predict with panel X and labels y."""

    _tags = {"X_univariate": True, "pre-refactor": True}

    args = {
        "fit": {"y": y, "X": X},
        "predict": {"X": X_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


scenarios_classification = [
    ClassifierFitPredict,
]

# we use the same scenarios for regression, as in the old test suite
scenarios_regression = [
    ClassifierFitPredict,
]


# old code for easy refactor

# def _make_args(estimator, method, **kwargs):
#     """Generate testing arguments for estimator methods."""
#     if method == "fit":
#         return _make_fit_args(estimator, **kwargs)
#     if method == "update":
#         raise NotImplementedError()
#     elif method in ("predict", "predict_proba", "decision_function"):
#         return _make_predict_args(estimator, **kwargs)
#     elif method == "transform":
#         return _make_transform_args(estimator, **kwargs)
#     elif method == "inverse_transform":
#         return _make_inverse_transform_args(estimator, **kwargs)
#     else:
#         raise ValueError(f"Method: {method} not supported")


# def _make_fit_args(estimator, **kwargs):
#     if isinstance(estimator, BaseForecaster):
#         # we need to handle the TransformedTargetForecaster separately
#         if isinstance(estimator, _SeriesToSeriesTransformer):
#             y = _make_series(**kwargs)
#         else:
#             # create matching n_columns input, if n_columns not passed
#             # e.g., to give bivariate y to strictly multivariate forecaster
#             if "n_columns" not in kwargs.keys():
#                 n_columns = _get_n_columns(
#                     estimator.get_tag(tag_name="scitype:y", raise_error=False)
#                 )[0]
#                 y = make_forecasting_problem(n_columns=n_columns, **kwargs)
#             else:
#                 y = make_forecasting_problem(**kwargs)
#         fh = 1
#         X = None
#         return y, X, fh
#     elif isinstance(estimator, BaseSeriesAnnotator):
#         X = make_annotation_problem(**kwargs)
#         return (X,)
#     elif isinstance(estimator, BaseClassifier):
#         return make_classification_problem(**kwargs)
#     elif isinstance(estimator, BaseRegressor):
#         return make_regression_problem(**kwargs)
#     elif isinstance(
#         estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
#     ):
#         X = _make_series(**kwargs)
#         return (X,)
#     elif isinstance(estimator, (_PanelToTabularTransformer,_PanelToPanelTransformer)):
#         return make_classification_problem(**kwargs)
#     elif isinstance(estimator, BaseTransformer):
#         X = _make_series(**kwargs)
#         return (X,)
#     elif isinstance(estimator, BaseClusterer):
#         return (make_clustering_problem(**kwargs),)
#     elif isinstance(estimator, BasePairwiseTransformer):
#         return None, None
#     elif isinstance(estimator, BasePairwiseTransformerPanel):
#         return None, None
#     elif isinstance(estimator, BaseAligner):
#         X = [_make_series(n_columns=2, **kwargs), _make_series(n_columns=2, **kwargs)]
#         return (X,)
#     else:
#         raise ValueError(_get_err_msg(estimator))


# def _make_predict_args(estimator, **kwargs):
#     if isinstance(estimator, BaseForecaster):
#         fh = 1
#         return (fh,)
#     elif isinstance(estimator, (BaseClassifier, BaseRegressor)):
#         X = _make_panel_X(**kwargs)
#         return (X,)
#     elif isinstance(estimator, BaseSeriesAnnotator):
#         X = make_annotation_problem(n_timepoints=10, **kwargs)
#         return (X,)
#     elif isinstance(estimator, BaseClusterer):
#         X = _make_panel_X(**kwargs)
#         return (X,)
#     else:
#         raise ValueError(_get_err_msg(estimator))
