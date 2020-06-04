__all__ = [
    "_construct_instance",
    "_make_args",
    "_assert_almost_equal"
]
__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
from sktime.classification.base import is_classifier
from sktime.forecasting.base import is_forecaster
from sktime.regression.base import is_regressor
from sktime.tests._config import ESTIMATOR_TEST_PARAMS
from sktime.transformers.series_as_features.base import \
    is_series_as_features_transformer
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.transformers.single_series.base import is_single_series_transformer
from sktime.utils.data_container import is_nested_dataframe
from sktime.utils.data_container import tabularize
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.series_as_features import \
    make_classification_problem
from sktime.utils._testing.series_as_features import make_regression_problem


def generate_df_from_array(array, n_rows=10, n_cols=1):
    return pd.DataFrame(
        [[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
        columns=[f'col{c}' for c in range(n_cols)])


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
                raise ValueError(f"Estimator: {Estimator} requires "
                                 f"non-default parameters for construction, "
                                 f"but none have been found")

        # construct with default parameters if none are required
        estimator = Estimator()

    return estimator


def _make_args(estimator, method, *args, **kwargs):
    """Helper function to generate appropriate arguments for testing different
    estimator types and their methods"""
    if method == "fit":
        return _make_fit_args(estimator, *args, **kwargs)

    elif method in ("predict", "predict_proba", "decision_function"):
        return _make_predict_args(estimator, *args, **kwargs)

    elif method == "transform":
        return _make_transform_args(estimator, *args, **kwargs)

    elif method == "inverse_transform":
        args = _make_transform_args(estimator, *args, **kwargs)
        if isinstance(estimator, Tabularizer):
            X, y = args
            return tabularize(X), y
        else:
            return args

    else:
        raise ValueError(f"Method: {method} not supported")


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


def _make_predict_args(estimator, *args, **kwargs):
    if is_forecaster(estimator):
        fh = 1
        return (fh,)

    elif is_classifier(estimator):
        X, y = make_classification_problem(*args, **kwargs)
        return (X,)

    elif is_regressor(estimator):
        X, y = make_regression_problem(*args, **kwargs)
        return (X,)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")


def _make_transform_args(estimator, random_state=None):
    if is_series_as_features_transformer(estimator):
        return make_classification_problem(random_state=random_state)

    elif is_single_series_transformer(estimator) or is_forecaster(estimator):
        y = make_forecasting_problem(random_state=random_state)
        return (y,)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")


def _assert_almost_equal(x, y, decimal=6, err_msg="", verbose=True):
    # we iterate over columns and rows make cell-wise comparisons,
    # tabularizing the data first would simplify this a bit, but does not
    # work for unequal length data

    if is_nested_dataframe(x):
        # make sure both inputs have the same shape
        if not x.shape == y.shape:
            raise ValueError("Found inputs with different shapes")

        # iterate over columns
        n_columns = x.shape[1]
        for i in range(n_columns):
            xc = x.iloc[:, i].tolist()
            yc = y.iloc[:, i].tolist()

            # iterate over rows, checking if cells are equal
            for xci, yci in zip(xc, yc):
                np.testing.assert_array_almost_equal(
                    xci, yci, decimal=decimal, err_msg=err_msg,
                    verbose=verbose)

    else:
        np.testing.assert_array_almost_equal(
            x, y, decimal=decimal, err_msg=err_msg, verbose=verbose)
