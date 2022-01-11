# -*- coding: utf-8 -*-
"""Test scenarios for pairwise transformers.

Contains TestScenario concrete children to run in tests for pairwise transformers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_transformers_pairwise", "scenarios_transformers_pairwise_panel"]

from inspect import isclass

import pandas as pd

from sktime.base import BaseObject
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


def get_tag(obj, tag_name):
    """Shorthand for get_tag vs get_class_tag, obj can be class or object."""
    if isclass(obj):
        return obj.get_class_tag(tag_name)
    else:
        return obj.get_tag(tag_name)


# no logic in scenario classes, but placeholder and for pattern homogeneity
class TransformerPairwiseTestScenario(TestScenario, BaseObject):
    """Generic test scenario for pairwise transformers."""

    pass


class TransformerPairwisePanelTestScenario(TestScenario, BaseObject):
    """Generic test scenario for pairwise panel transformers."""

    pass


d = {"col1": [1, 2], "col2": [3, 4]}
d = pd.DataFrame(d)

d2 = {"col1": [2, 3, 4], "col2": [3, 4, 5]}
d2 = pd.DataFrame(d2)


class TransformerPairwiseTransformSymm(TransformerPairwiseTestScenario):
    """Empty fit, one argument in transform."""

    _tags = {"symmetric": True, "pre-refactor": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": d},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwiseTransformAsymm(TransformerPairwiseTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "pre-refactor": False}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": d, "X2": d2},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers_pairwise = [
    TransformerPairwiseTransformSymm,
    TransformerPairwiseTransformAsymm,
]

X = [d, d]
X2 = [d2, d, d2]


class TransformerPairwisePanelTransformSymm(TransformerPairwisePanelTestScenario):
    """Empty fit, one argument in transform."""

    _tags = {"symmetric": True, "pre-refactor": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwisePanelTransformAsymm(TransformerPairwisePanelTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "pre-refactor": False}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X, "X2": X2},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers_pairwise_panel = [
    TransformerPairwisePanelTransformSymm,
    TransformerPairwisePanelTransformAsymm,
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


# def _make_transform_args(estimator, **kwargs):
#     if isinstance(
#         estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
#     ):
#         X = _make_series(**kwargs)
#         return (X,)
#     elif isinstance(
#         estimator,
#         (
#             _PanelToTabularTransformer,
#             _PanelToPanelTransformer,
#         ),
#     ):
#         X = _make_panel_X(**kwargs)
#         return (X,)
#     elif isinstance(estimator, BaseTransformer):
#         X = _make_series(**kwargs)
#         return (X,)
#     elif isinstance(estimator, BasePairwiseTransformer):
#         d = {"col1": [1, 2], "col2": [3, 4]}
#         return pd.DataFrame(d), pd.DataFrame(d)
#     elif isinstance(estimator, BasePairwiseTransformerPanel):
#         d = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
#         X = [d, d]
#         return X, X
#     else:
#         raise ValueError(_get_err_msg(estimator))


# def _make_inverse_transform_args(estimator, **kwargs):
#     if isinstance(estimator, _SeriesToPrimitivesTransformer):
#         X = _make_primitives(**kwargs)
#         return (X,)
#     elif isinstance(estimator, _SeriesToSeriesTransformer):
#         X = _make_series(**kwargs)
#         return (X,)
#     elif isinstance(estimator, _PanelToTabularTransformer):
#         X = _make_tabular_X(**kwargs)
#         return (X,)
#     elif isinstance(estimator, _PanelToPanelTransformer):
#         X = _make_panel_X(**kwargs)
#         return (X,)
#     elif isinstance(estimator, BaseTransformer):
#         X = _make_series(**kwargs)
#         return (X,)
#     else:
#         raise ValueError(_get_err_msg(estimator))
