__all__ = [
    "_construct_instance",
    "_make_args"
]
__author__ = ["Markus Löning"]

import pandas as pd
from sktime.classification.base import is_classifier
from sktime.forecasting.base import is_forecaster
from sktime.regression.base import is_regressor
from sktime.utils.testing.config import TEST_CONSTRUCT_CONFIG_LOOKUP
from sktime.utils.testing.forecasting import make_forecasting_problem
from sktime.utils.testing.series_as_features import make_classification_problem
from sktime.utils.testing.series_as_features import make_regression_problem


def generate_df_from_array(array, n_rows=10, n_cols=1):
    return pd.DataFrame(
        [[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
        columns=[f'col{c}' for c in range(n_cols)])


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""

    # some estimators require parameters during construction
    required_parameters = getattr(Estimator, "_required_parameters", [])

    # construct with test parameters
    if Estimator in TEST_CONSTRUCT_CONFIG_LOOKUP:
        params = TEST_CONSTRUCT_CONFIG_LOOKUP[Estimator]
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

    elif method in ("predict", "predict_proba"):
        return _make_predict_args(estimator, *args, **kwargs)

    else:
        raise ValueError(f"Method: {method} not supported")


def _make_fit_args(estimator, random_state=None):
    if is_forecaster(estimator):
        y = make_forecasting_problem(random_state=random_state)
        fh = 1
        return y, fh

    elif is_classifier(estimator):
        return make_classification_problem(random_state=random_state)

    elif is_regressor(estimator):
        return make_regression_problem(random_state=random_state)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")


def _make_predict_args(estimator, random_state=None):
    if is_forecaster(estimator):
        fh = 1
        return (fh,)

    elif is_classifier(estimator):
        X, y = make_classification_problem(random_state=random_state)
        return (X,)

    elif is_regressor(estimator):
        X, y = make_regression_problem(random_state=random_state)
        return (X,)

    else:
        raise ValueError(f"Estimator type: {type(estimator)} not supported")
