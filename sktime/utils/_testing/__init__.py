# -*- coding: utf-8 -*-
__all__ = [
    "_construct_instance",
    "_make_args",
    "_assert_array_almost_equal",
    "_assert_array_equal",
]
__author__ = ["Markus Löning"]

from inspect import signature

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from sktime.classification.base import is_classifier
from sktime.forecasting.base._base import is_forecaster
from sktime.regression.base import is_regressor
from sktime.tests._config import ESTIMATOR_TEST_PARAMS
from sktime.transformers.series_as_features.base import (
    is_series_as_features_transformer,
)
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.transformers.single_series.base import is_single_series_transformer
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.series_as_features import make_classification_problem
from sktime.utils._testing.series_as_features import make_regression_problem
from sktime.utils.data_container import from_3d_numpy_to_2d_array
from sktime.utils.data_container import from_nested_to_2d_array
from sktime.utils.data_container import is_nested_dataframe


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""

    # construct with parameter configuration for testing
    if Estimator in ESTIMATOR_TEST_PARAMS:
        params = ESTIMATOR_TEST_PARAMS[Estimator]
        estimator = Estimator(**params)

    # otherwise construct with default parameters
    else:
        # if non-default parameters are required, but none have been found,
        # raise error
        if hasattr(Estimator, "_required_parameters"):
            required_parameters = getattr(Estimator, "required_parameters", [])
            if len(required_parameters) > 0:
                raise ValueError(
                    f"Estimator: {Estimator} requires "
                    f"non-default parameters for construction, "
                    f"but none have been found"
                )

        # construct with default parameters if none are required
        estimator = Estimator()

    return estimator


def _make_args(estimator, method, **kwargs):
    """Helper function to generate appropriate arguments for testing different
    estimator types and their methods"""
    if method == "fit":
        return _make_fit_args(estimator, **kwargs)

    elif method in ("predict", "predict_proba", "decision_function"):
        return _make_predict_args(estimator, **kwargs)

    elif method == "transform":
        return _make_transform_args(estimator, **kwargs)

    elif method == "inverse_transform":
        args = _make_transform_args(estimator, **kwargs)

        if isinstance(estimator, Tabularizer):
            args = _handle_tabularizer_args(*args, **kwargs)

        return args

    else:
        raise ValueError(f"Method: {method} not supported")


def _handle_tabularizer_args(*args, **kwargs):
    # the Tabularizer transforms a nested pd.DataFrame/3d numpy array into a
    # 2d numpy array, so the inverse transform goes from a 2d numpy array to a
    # nested pd.DataFrame/3d array
    # TODO refactor Tabularizer as series-as-features composition meta-estimator,
    #  rather than transformer or introduce special transformer type
    X, y = args
    if "return_numpy" in kwargs and kwargs["return_numpy"]:
        return from_3d_numpy_to_2d_array(X), y
    else:
        return from_nested_to_2d_array(X), y


def _make_fit_args(estimator, random_state=None, **kwargs):
    if is_forecaster(estimator):
        y = make_forecasting_problem(random_state=random_state, **kwargs)
        fh = 1
        return y, fh

    elif is_classifier(estimator):
        return make_classification_problem(random_state=random_state, **kwargs)

    elif is_regressor(estimator):
        return make_regression_problem(random_state=random_state, **kwargs)

    elif is_series_as_features_transformer(estimator):
        return make_classification_problem(random_state=random_state, **kwargs)

    elif is_single_series_transformer(estimator):
        y = make_forecasting_problem(random_state=random_state, **kwargs)
        return (y,)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")


def _make_predict_args(estimator, **kwargs):
    if is_forecaster(estimator):
        fh = 1
        return (fh,)

    elif is_classifier(estimator):
        X, y = make_classification_problem(**kwargs)
        return (X,)

    elif is_regressor(estimator):
        X, y = make_regression_problem(**kwargs)
        return (X,)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")


def _make_transform_args(estimator, return_numpy=False, random_state=None):
    if is_series_as_features_transformer(estimator):
        return make_classification_problem(
            return_numpy=return_numpy, random_state=random_state
        )

    elif is_single_series_transformer(estimator) or is_forecaster(estimator):
        y = make_forecasting_problem(random_state=random_state)
        return (y,)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")


def _compare_nested_frame(func, x, y, **kwargs):
    """Helper function to compare two nested pd.DataFrames"""
    # we iterate over columns and rows to make cell-wise comparisons,
    # tabularizing the data first would simplify this, but does not
    # work for unequal length data

    # in some cases, x and y may be empty (e.g. TSFreshRelevantFeatureExtractor) and
    # we cannot compare individual cells, so we simply check if they are equal
    assert isinstance(x, pd.DataFrame)
    if x.empty:
        assert_frame_equal(x, y)

    elif is_nested_dataframe(x):
        # make sure both inputs have the same shape
        if not x.shape == y.shape:
            raise ValueError("Found inputs with different shapes")

        # iterate over columns, checking individuals cells
        n_columns = x.shape[1]
        for i in range(n_columns):
            xc = x.iloc[:, i].tolist()
            yc = y.iloc[:, i].tolist()

            # iterate over rows, checking if cells are equal
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
    """Helper to get function arguments"""
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
