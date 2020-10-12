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
from sklearn.utils.validation import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.forecasting.base._base import BaseForecaster
from sktime.regression.base import BaseRegressor
from sktime.tests._config import ESTIMATOR_TEST_PARAMS
from sktime.tests._config import VALID_ESTIMATOR_TAGS
from sktime.tests._config import VALID_ESTIMATOR_TYPES
from sktime.transformers.base import _SeriesAsFeaturesToSeriesAsFeaturesTransformer
from sktime.transformers.base import _SeriesAsFeaturesToTabularTransformer
from sktime.transformers.base import _SeriesToPrimitivesTransformer
from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.series_as_features import _make_series_as_features_X
from sktime.utils._testing.series_as_features import make_classification_problem
from sktime.utils._testing.series_as_features import make_regression_problem
from sktime.utils.data_container import is_nested_dataframe


def _get_err_msg(estimator):
    return (
        f"Invalid estimator type: {type(estimator)}. Valid estimator types are: "
        f"{VALID_ESTIMATOR_TYPES}"
    )


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
    if method == "update":
        raise NotImplementedError()
    elif method in ("predict", "predict_proba", "decision_function"):
        return _make_predict_args(estimator, **kwargs)
    elif method == "transform":
        return _make_transform_args(estimator, **kwargs)
    elif method == "inverse_transform":
        return _make_inverse_transform_args(estimator, **kwargs)
    elif method == "fit-transform":
        return _make_fit_transform_args(estimator, **kwargs)
    else:
        raise ValueError(f"Method: {method} not supported")


def _make_fit_transform_args(estimator, **kwargs):
    # For forecasters which are also transformers (e.g. pipelines), we cannot
    # the forecasting horizon to transform, so we only generate a series here. Note
    # that this will fail for forecasters which require the forecasting horizon in fit.
    if isinstance(estimator, BaseForecaster) and isinstance(
        estimator, _SeriesToSeriesTransformer
    ):
        y = _make_series(**kwargs)
        return (y,)
    else:
        return _make_fit_args(estimator, **kwargs)


def _make_fit_args(estimator, **kwargs):
    if isinstance(estimator, BaseForecaster):
        y = make_forecasting_problem(**kwargs)
        fh = 1
        X = None
        return y, X, fh
    elif isinstance(estimator, BaseClassifier):
        return make_classification_problem(**kwargs)
    elif isinstance(estimator, BaseRegressor):
        return make_regression_problem(**kwargs)
    elif isinstance(
        estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
    ):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(
        estimator,
        (
            _SeriesAsFeaturesToTabularTransformer,
            _SeriesAsFeaturesToSeriesAsFeaturesTransformer,
        ),
    ):
        return make_classification_problem(**kwargs)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_predict_args(estimator, **kwargs):
    if isinstance(estimator, BaseForecaster):
        fh = 1
        return (fh,)
    elif isinstance(estimator, (BaseClassifier, BaseRegressor)):
        X = _make_series_as_features_X(**kwargs)
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
            _SeriesAsFeaturesToTabularTransformer,
            _SeriesAsFeaturesToSeriesAsFeaturesTransformer,
        ),
    ):
        X = _make_series_as_features_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_inverse_transform_args(estimator, **kwargs):
    if isinstance(estimator, _SeriesToPrimitivesTransformer):
        X = _make_primitives(**kwargs)
        return (X,)
    elif isinstance(estimator, _SeriesToSeriesTransformer):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(estimator, _SeriesAsFeaturesToTabularTransformer):
        X = _make_tabular_X(**kwargs)
        return (X,)
    elif isinstance(estimator, _SeriesAsFeaturesToSeriesAsFeaturesTransformer):
        X = _make_series_as_features_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_primitives(n_columns=1, random_state=None):
    """Generate one or more primitives. Useful for checking inverse-transform
    of series-to-primitives transformer"""
    rng = check_random_state(random_state)
    if n_columns == 1:
        return rng.rand()
    return np.rand(size=(n_columns,))


def _make_tabular_X(n_instances=20, n_columns=1, return_numpy=True, random_state=None):
    """Generate tabular X. Useful for checking inverse-transform
    of series-as-features-to-tabular transformer"""
    rng = check_random_state(random_state)
    data = rng.rand(n_instances, n_columns)
    if return_numpy:
        return data
    else:
        return pd.DataFrame(data)


def _compare_nested_frame(func, x, y, **kwargs):
    """Helper function to compare two nested pd.DataFrames

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


def _has_tag(Estimator, tag):
    assert tag in VALID_ESTIMATOR_TAGS
    # Check if tag is in all tags
    return Estimator._all_tags().get(tag, False)
