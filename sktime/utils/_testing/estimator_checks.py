"""Utility function for estimator testing.

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["mloning", "fkiraly"]

from inspect import isclass, signature

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.utils.validation import check_random_state

from sktime.alignment.base import BaseAligner
from sktime.base import BaseEstimator, BaseObject
from sktime.classification.base import BaseClassifier
from sktime.classification.early_classification import BaseEarlyClassifier
from sktime.clustering.base import BaseClusterer
from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.dists_kernels import BasePairwiseTransformer, BasePairwiseTransformerPanel
from sktime.forecasting.base import BaseForecaster
from sktime.regression.base import BaseRegressor
from sktime.tests._config import VALID_ESTIMATOR_TYPES
from sktime.transformations.base import BaseTransformer


def _get_err_msg(estimator):
    return (
        f"Invalid estimator type: {type(estimator)}. Valid estimator types are: "
        f"{VALID_ESTIMATOR_TYPES}"
    )


def _list_required_methods(estimator):
    """Return list of required method names (beyond BaseEstimator ones)."""
    # all BaseObject children must implement these
    MUST_HAVE_FOR_OBJECTS = ["set_params", "get_params"]

    # all BaseEstimator children must implement these
    MUST_HAVE_FOR_ESTIMATORS = [
        "fit",
        "check_is_fitted",
        "is_fitted",  # read-only property
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

    if isinstance(estimator, BaseObject):
        required_methods += MUST_HAVE_FOR_OBJECTS

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


def _has_capability(est, method: str) -> bool:
    """Check whether estimator has capability of method."""

    def get_tag(est, tag_name, tag_value_default=None):
        if isclass(est):
            return est.get_class_tag(
                tag_name=tag_name, tag_value_default=tag_value_default
            )
        else:
            return est.get_tag(
                tag_name=tag_name,
                tag_value_default=tag_value_default,
                raise_error=False,
            )

    if not hasattr(est, method):
        return False
    if method == "inverse_transform":
        return get_tag(est, "capability:inverse_transform", False)
    if method in [
        "predict_proba",
        "predict_interval",
        "predict_quantiles",
        "predict_var",
    ]:
        ALWAYS_HAVE_PREDICT_PROBA = (BaseClassifier, BaseEarlyClassifier, BaseClusterer)
        # all classifiers and clusterers implement predict_proba
        if method == "predict_proba" and isinstance(est, ALWAYS_HAVE_PREDICT_PROBA):
            return True
        return get_tag(est, "capability:pred_int", False)
    # skip transform for forecasters that have it - pipelines
    if method == "transform" and isinstance(est, BaseForecaster):
        return False
    return True
