# -*- coding: utf-8 -*-
"""Test scenarios for transformers.

Contains TestScenario concrete children to run in tests for transformers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_transformers"]


from inspect import isclass

from sktime.base import BaseObject
from sktime.transformations.base import (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils._testing.scenarios import TestScenario

OLD_MIXINS = (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)

OLD_PANEL_MIXINS = (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
)

OLD_SERIES_MIXINS = (
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)


def _is_child_of(obj, class_or_tuple):
    """Shorthand for 'inherits from', obj can be class or object."""
    if isclass(obj):
        return issubclass(obj, class_or_tuple)
    else:
        return isinstance(obj, class_or_tuple)


class TransformerTestScenario(TestScenario, BaseObject):
    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """

        def get_tag(obj, tag_name):
            if isclass(obj):
                return obj.get_class_tag(tag_name)
            else:
                return obj.get_tag(tag_name)

        # pre-refactor classes can't deal with Series *and* Panel both
        X_scitype = self.get_tag("X_scitype")

        if _is_child_of(obj, OLD_PANEL_MIXINS) and X_scitype != "Panel":
            return False

        if _is_child_of(obj, OLD_SERIES_MIXINS) and X_scitype != "Series":
            return False

        # applicable only if number of variables in y complies with scitype:y
        is_univariate = self.get_tag("X_univariate")

        if not is_univariate and get_tag(obj, "univariate-only"):
            return False

        return True


class TransformerFitTransformSeriesUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Series X."""

    _tags = {"X_scitype": "Series", "X_univariate": True}

    args = {
        "fit": {"X": _make_series(n_timepoints=20)},
        "transform": {"X": _make_series(n_timepoints=10)},
        # "inverse_transform": {"X": _make_series(n_timepoints=10)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformSeriesMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Series X."""

    _tags = {"X_scitype": "Series", "X_univariate": True}

    args = {
        "fit": {"X": _make_series(n_columns=2, n_timepoints=20)},
        "transform": {"X": _make_series(n_columns=2, n_timepoints=10)},
        # "inverse_transform": {"X": _make_series(n_timepoints=10)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Panel X."""

    _tags = {"X_scitype": "Panel", "X_univariate": True}

    args = {
        "fit": {"X": _make_panel_X(n_instances=7, n_columns=1, n_timepoints=20)},
        "transform": {"X": _make_panel_X(n_instances=3, n_columns=1, n_timepoints=10)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Panel X."""

    _tags = {"X_scitype": "Panel", "X_univariate": False}

    args = {
        "fit": {"X": _make_panel_X(n_instances=7, n_columns=2, n_timepoints=20)},
        "transform": {"X": _make_panel_X(n_instances=3, n_columns=2, n_timepoints=10)},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers = [
    TransformerFitTransformSeriesUnivariate,
    TransformerFitTransformSeriesMultivariate,
    TransformerFitTransformPanel,
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
