# -*- coding: utf-8 -*-
"""Test scenarios for transformers.

Contains TestScenario concrete children to run in tests for transformers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_transformers"]


from inspect import isclass

from sktime.base import BaseObject
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.scenarios import TestScenario


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

        return True

        # # applicable only if obj inherits from BaseForecaster
        # applicable = isinstance(obj, BaseForecaster) or issubclass(obj,BaseForecaster)

        # # applicable only if number of variables in y complies with scitype:y
        # is_univariate = self.get_tag("univariate_y")

        # if is_univariate and get_tag(obj, "scitype:y") == "multivariate":
        #     applicable = False

        # if not is_univariate and get_tag(obj, "scitype:y") == "univariate":
        #     applicable = False

        # # applicable only if fh is not passed later than it needs to be
        # fh_in_fit = self.get_tag("fh_passed_in_fit")

        # if not fh_in_fit and get_tag(obj, "requires-fh-in-fit"):
        #     applicable = False

        # return applicable


class TransformerFitTransform(TransformerTestScenario):
    """Fit/predict only, univariate y, no X."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    args = {"fit": {"y": _make_series(), "fh": 1}, "predict": {"fh": 1}}
    default_method_sequence = ["fit", "predict"]


scenarios_transformers = [TransformerFitTransform]


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
