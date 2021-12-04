# -*- coding: utf-8 -*-
"""Utility function for estimator testing.

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["mloning", "fkiraly"]

from inspect import signature

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state

from sktime.alignment.base import BaseAligner
from sktime.annotation.base import BaseSeriesAnnotator
from sktime.classification.base import BaseClassifier
from sktime.clustering.base.base import BaseClusterer
from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.dists_kernels import BasePairwiseTransformer, BasePairwiseTransformerPanel
from sktime.forecasting.base import BaseForecaster
from sktime.regression.base import BaseRegressor
from sktime.tests._config import VALID_ESTIMATOR_TYPES
from sktime.transformations.base import (
    BaseTransformer,
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)
from sktime.utils._testing.annotation import make_annotation_problem
from sktime.utils._testing.forecasting import (
    _get_n_columns,
    _make_series,
    make_forecasting_problem,
)
from sktime.utils._testing.panel import (
    _make_panel_X,
    make_classification_problem,
    make_clustering_problem,
    make_regression_problem,
)


def _get_err_msg(estimator):
    return (
        f"Invalid estimator type: {type(estimator)}. Valid estimator types are: "
        f"{VALID_ESTIMATOR_TYPES}"
    )


def _construct_instance(Estimator):
    """Construct Estimator instance if possible."""
    # return the instance of the class with default parameters
    return Estimator.create_test_instance()


def _list_required_methods(estimator):
    """Return list of required method names (beyond BaseEstimator ones)."""
    # all BaseEstimator children must implement these
    MUST_HAVE_FOR_ESTIMATORS = [
        "fit",
        "check_is_fitted",
        "is_fitted",  # read-only property
        "set_params",
        "get_params",
    ]
    # prediction/forecasting base classes that must have predict
    BASE_CLASSES_THAT_MUST_HAVE_PREDICT = (
        BaseClusterer,
        BaseRegressor,
        BaseForecaster,
    )
    # transformation base classes that must have transform
    BASE_CLASSES_THAT_MUST_HAVE_TRANSFORM = (
        BaseTransformer,
        BasePairwiseTransformer,
        BasePairwiseTransformerPanel,
    )

    required_methods = []

    if isinstance(estimator, BaseEstimator):
        required_methods += MUST_HAVE_FOR_ESTIMATORS

    if isinstance(estimator, BASE_CLASSES_THAT_MUST_HAVE_PREDICT):
        required_methods += ["predict"]

    if isinstance(estimator, BASE_CLASSES_THAT_MUST_HAVE_TRANSFORM):
        required_methods += ["transform"]

    if isinstance(estimator, BaseAligner):
        required_methods += [
            "get_alignment",
            "get_alignment_loc",
            "get_aligned",
            "get_distance",
            "get_distance_matrix",
        ]

    return required_methods


def _make_args(estimator, method, **kwargs):
    """Generate testing arguments for estimator methods."""
    if method == "fit":
        return _make_fit_args(estimator, **kwargs)
    if method == "update":
        raise NotImplementedError()
    elif method in ("predict", "predict_proba", "decision_function"):
        return _make_predict_args(estimator, **kwargs)
    elif method == "transform":
        return _make_transform_args(estimator, **kwargs)
    elif method == "inverse_transform":
        return _make_inverse_transform_args(estimator, **kwargs)
    else:
        raise ValueError(f"Method: {method} not supported")


def _make_fit_args(estimator, **kwargs):
    if isinstance(estimator, BaseForecaster):
        # we need to handle the TransformedTargetForecaster separately
        if isinstance(estimator, _SeriesToSeriesTransformer):
            y = _make_series(**kwargs)
        else:
            # create matching n_columns input, if n_columns not passed
            # e.g., to give bivariate y to strictly multivariate forecaster
            if "n_columns" not in kwargs.keys():
                n_columns = _get_n_columns(
                    estimator.get_tag(tag_name="scitype:y", raise_error=False)
                )[0]
                y = make_forecasting_problem(n_columns=n_columns, **kwargs)
            else:
                y = make_forecasting_problem(**kwargs)
        fh = 1
        X = None
        return y, X, fh
    elif isinstance(estimator, BaseSeriesAnnotator):
        X = make_annotation_problem(**kwargs)
        return (X,)
    elif isinstance(estimator, BaseClassifier):
        return make_classification_problem(**kwargs)
    elif isinstance(estimator, BaseRegressor):
        return make_regression_problem(**kwargs)
    elif isinstance(
        estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
    ):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(estimator, (_PanelToTabularTransformer, _PanelToPanelTransformer)):
        return make_classification_problem(**kwargs)
    elif isinstance(estimator, BaseClusterer):
        return (make_clustering_problem(**kwargs),)
    elif isinstance(estimator, BasePairwiseTransformer):
        return None, None
    elif isinstance(estimator, BasePairwiseTransformerPanel):
        return None, None
    elif isinstance(estimator, BaseAligner):
        X = [_make_series(n_columns=2, **kwargs), _make_series(n_columns=2, **kwargs)]
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_predict_args(estimator, **kwargs):
    if isinstance(estimator, BaseForecaster):
        fh = 1
        return (fh,)
    elif isinstance(estimator, (BaseClassifier, BaseRegressor)):
        X = _make_panel_X(**kwargs)
        return (X,)
    elif isinstance(estimator, BaseSeriesAnnotator):
        X = make_annotation_problem(n_timepoints=10, **kwargs)
        return (X,)
    elif isinstance(estimator, BaseClusterer):
        X = _make_panel_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_transform_args(estimator, **kwargs):
    if isinstance(
        estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
    ):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(
        estimator,
        (
            _PanelToTabularTransformer,
            _PanelToPanelTransformer,
        ),
    ):
        X = _make_panel_X(**kwargs)
        return (X,)
    elif isinstance(estimator, BasePairwiseTransformer):
        d = {"col1": [1, 2], "col2": [3, 4]}
        return pd.DataFrame(d), pd.DataFrame(d)
    elif isinstance(estimator, BasePairwiseTransformerPanel):
        d = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        X = [d, d]
        return X, X
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_inverse_transform_args(estimator, **kwargs):
    if isinstance(estimator, _SeriesToPrimitivesTransformer):
        X = _make_primitives(**kwargs)
        return (X,)
    elif isinstance(estimator, _SeriesToSeriesTransformer):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(estimator, _PanelToTabularTransformer):
        X = _make_tabular_X(**kwargs)
        return (X,)
    elif isinstance(estimator, _PanelToPanelTransformer):
        X = _make_panel_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_primitives(n_columns=1, random_state=None):
    """Generate one or more primitives, for checking inverse-transform."""
    rng = check_random_state(random_state)
    if n_columns == 1:
        return rng.rand()
    return rng.rand(size=(n_columns,))


def _make_tabular_X(n_instances=20, n_columns=1, return_numpy=True, random_state=None):
    """Generate tabular X, for checking inverse-transform."""
    rng = check_random_state(random_state)
    X = rng.rand(n_instances, n_columns)
    if return_numpy:
        return X
    else:
        return pd.DataFrame(X)


def _compare_nested_frame(func, x, y, **kwargs):
    """Compare two nested pd.DataFrames.

    Parameters
    ----------
    func : function
        Function from np.testing for comparing arrays.
    x : pd.DataFrame
    y : pd.DataFrame
    kwargs : dict
        Keyword argument for function

    Raises
    ------
    AssertionError
        If x and y are not equal
    """
    # We iterate over columns and rows to make cell-wise comparisons.
    # Tabularizing the data first would simplify this, but does not
    # work for unequal length data.

    # In rare cases, x and y may be empty (e.g. TSFreshRelevantFeatureExtractor) and
    # we cannot compare individual cells, so we simply check if everything else is
    # equal here.
    assert isinstance(x, pd.DataFrame)
    if x.empty:
        assert_frame_equal(x, y)

    elif is_nested_dataframe(x):
        # Check if both inputs have the same shape
        if not x.shape == y.shape:
            raise ValueError("Found inputs with different shapes")

        # Iterate over columns
        n_columns = x.shape[1]
        for i in range(n_columns):
            xc = x.iloc[:, i].tolist()
            yc = y.iloc[:, i].tolist()

            # Iterate over rows, checking if individual cells are equal
            for xci, yci in zip(xc, yc):
                func(xci, yci, **kwargs)


def _assert_array_almost_equal(x, y, decimal=6, err_msg=""):
    func = np.testing.assert_array_almost_equal
    if isinstance(x, pd.DataFrame):
        _compare_nested_frame(func, x, y, decimal=decimal, err_msg=err_msg)
    else:
        func(x, y, decimal=decimal, err_msg=err_msg)


def _assert_array_equal(x, y, err_msg=""):
    func = np.testing.assert_array_equal
    if isinstance(x, pd.DataFrame):
        _compare_nested_frame(func, x, y, err_msg=err_msg)
    else:
        func(x, y, err_msg=err_msg)


def _get_args(function, varargs=False):
    """Get function arguments."""
    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [
        key
        for key, param in params.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    if varargs:
        varargs = [
            param.name
            for param in params.values()
            if param.kind == param.VAR_POSITIONAL
        ]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args
