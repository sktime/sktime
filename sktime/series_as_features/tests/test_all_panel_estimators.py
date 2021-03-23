#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "test_classifier_output",
    "test_regressor_output",
    "test_multivariate_input",
    "test_3d_numpy_input",
]

import numpy as np
import pytest

from sktime.series_as_features.tests._config import ACCEPTED_OUTPUT_TYPES
from sktime.tests._config import EXCLUDE_ESTIMATORS
from sktime.tests._config import NON_STATE_CHANGING_METHODS
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils import all_estimators
from sktime.utils._testing.estimator_checks import _construct_instance
from sktime.utils._testing.estimator_checks import _make_args

CLASSIFIERS = all_estimators(
    "classifier", return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS
)
REGRESSORS = all_estimators(
    "regressor", return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS
)

PANEL_TRANSFORMERS = all_estimators(
    estimator_types=[_PanelToPanelTransformer, _PanelToTabularTransformer],
    return_names=False,
    exclude_estimators=EXCLUDE_ESTIMATORS,
)

PANEL_ESTIMATORS = CLASSIFIERS + REGRESSORS + PANEL_TRANSFORMERS

# We here only check the ouput for a single number of classes
N_CLASSES = 3


@pytest.mark.parametrize("Estimator", PANEL_ESTIMATORS)
def test_3d_numpy_input(Estimator):
    estimator = _construct_instance(Estimator)
    fit_args = _make_args(estimator, "fit", return_numpy=True)
    estimator.fit(*fit_args)

    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):

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


@pytest.mark.parametrize("Estimator", CLASSIFIERS + REGRESSORS)
def test_multivariate_input(Estimator):
    # check if multivariate input is correctly handled
    n_columns = 2
    error_msg = (
        f"X must be univariate "
        f"with X.shape[1] == 1, but found: "
        f"X.shape[1] == {n_columns}."
    )

    estimator = _construct_instance(Estimator)
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
    estimator = _construct_instance(Estimator)
    X_train, y_train = _make_args(estimator, "fit", n_classes=N_CLASSES)
    estimator.fit(X_train, y_train)

    X_new = _make_args(estimator, "predict")[0]

    # check predict
    y_pred = estimator.predict(X_new)
    assert isinstance(y_pred, ACCEPTED_OUTPUT_TYPES)
    assert y_pred.shape == (X_new.shape[0],)
    assert np.all(np.isin(np.unique(y_pred), np.unique(y_train)))

    # check predict proba
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_new)
        assert isinstance(y_proba, ACCEPTED_OUTPUT_TYPES)
        assert y_proba.shape == (X_new.shape[0], N_CLASSES)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize("Estimator", REGRESSORS)
def test_regressor_output(Estimator):
    estimator = _construct_instance(Estimator)
    X_train, y_train = _make_args(estimator, "fit")
    estimator.fit(X_train, y_train)

    X_new = _make_args(estimator, "predict")[0]

    # check predict
    y_pred = estimator.predict(X_new)
    assert isinstance(y_pred, ACCEPTED_OUTPUT_TYPES)
    assert y_pred.shape == (X_new.shape[0],)
    assert np.issubdtype(y_pred.dtype, np.floating)
