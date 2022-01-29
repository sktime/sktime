# -*- coding: utf-8 -*-
"""Unit tests for classifier/regressor input output."""

__author__ = ["mloning", "TonyBagnall"]
__all__ = [
    "test_classifier_output",
    "test_multivariate_input",
    "test_3d_numpy_input",
]

import numpy as np
import pytest

from sktime.registry import all_estimators
from sktime.tests._config import EXCLUDE_ESTIMATORS, NON_STATE_CHANGING_METHODS
from sktime.utils._testing.estimator_checks import _has_capability, _make_args

CLASSIFIERS = all_estimators(
    "classifier", return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS
)
N_CLASSES = 3


@pytest.mark.parametrize("Estimator", CLASSIFIERS)
def test_3d_numpy_input(Estimator):
    """Test classifiers handle 3D numpy input correctly."""
    estimator = Estimator.create_test_instance()
    fit_args = _make_args(estimator, "fit", return_numpy=True)
    estimator.fit(*fit_args)

    for method in NON_STATE_CHANGING_METHODS:
        if _has_capability(estimator, method):

            # try if methods can handle 3d numpy input data
            try:
                args = _make_args(estimator, method, return_numpy=True)
                getattr(estimator, method)(*args)

            # if not, check if they raise the appropriate error message
            except ValueError as e:
                error_msg = "This method requires X to be a nested pd.DataFrame"
                assert error_msg in str(e), (
                    f"{estimator.__class__.__name__} does "
                    f"not handle 3d numpy input data correctly"
                )


@pytest.mark.parametrize("Estimator", CLASSIFIERS)
def test_multivariate_input(Estimator):
    """Test classifiers handle multivariate pd.DataFrame input correctly."""
    # check if multivariate input is correctly handled
    n_columns = 2
    error_msg = "X must be univariate"

    estimator = Estimator.create_test_instance()
    X_train, y_train = _make_args(estimator, "fit", n_columns=n_columns)

    # check if estimator can handle multivariate data
    try:
        estimator.fit(X_train, y_train)
        for method in ("predict", "predict_proba"):
            X = _make_args(estimator, method, n_columns=n_columns)[0]
            getattr(estimator, method)(X)

    # if not, check if we raise error with appropriate message
    except ValueError as e:
        assert error_msg in str(e), (
            f"{estimator.__class__.__name__} does not handle multivariate "
            f"data and does not raise an appropriate error when multivariate "
            f"data is passed"
        )


@pytest.mark.parametrize("Estimator", CLASSIFIERS)
def test_classifier_output(Estimator):
    """Test classifier outputs the correct data types and values.

    Test predict produces a np.array or pd.Series with only values seen in the train
    data, and that predict_proba probability estimates add up to one.
    """
    estimator = Estimator.create_test_instance()
    X_train, y_train = _make_args(estimator, "fit", n_classes=N_CLASSES)
    estimator.fit(X_train, y_train)

    X_new = _make_args(estimator, "predict")[0]

    # check predict
    y_pred = estimator.predict(X_new)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X_new.shape[0],)
    assert np.all(np.isin(np.unique(y_pred), np.unique(y_train)))

    # check predict proba
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_new)
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (X_new.shape[0], N_CLASSES)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1)
