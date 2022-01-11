# -*- coding: utf-8 -*-
"""Test scenarios for clustering.

Contains TestScenario concrete children to run in tests for clusterers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_clustering"]


from sktime.base import BaseObject
from sktime.utils._testing.panel import _make_panel_X, make_clustering_problem
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


# no logic in scenario classes, but placeholder and for pattern homogeneity
class ClustererTestScenario(TestScenario, BaseObject):
    """Generic test scenario for clusterers."""

    pass


class ClustererFitPredict(ClustererTestScenario):
    """Fit/predict with panel Xmake_clustering_problem."""

    _tags = {"X_univariate": True, "pre-refactor": True}

    args = {
        "fit": {"X": make_clustering_problem(random_state=RAND_SEED)},
        "predict": {"X": _make_panel_X(random_state=RAND_SEED)},
    }
    default_method_sequence = ["fit", "predict"]


scenarios_clustering = [
    ClustererFitPredict,
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
