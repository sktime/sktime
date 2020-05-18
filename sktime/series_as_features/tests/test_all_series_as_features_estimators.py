#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd
import pytest
from sktime.tests.config import EXCLUDED
from sktime.utils import all_estimators
from sktime.utils.testing import _construct_instance
from sktime.utils.testing import _make_args

ALL_CLASSIFIERS = [e[1] for e in
                   all_estimators(estimator_type="classifier")
                   if e[0] not in EXCLUDED]

ALL_REGRESSORS = [e[1] for e in
                  all_estimators(estimator_type="regressor")
                  if e[0] not in EXCLUDED]

N_CLASSES = 3
ACCEPTED_OUTPUT_TYPES = (np.ndarray, pd.Series)


@pytest.mark.parametrize("Estimator", ALL_CLASSIFIERS)
def test_classifier_output(Estimator):
    estimator = _construct_instance(Estimator)
    X_train, y_train = _make_args(estimator, "fit", n_classes=N_CLASSES)
    estimator.fit(X_train, y_train)

    X = _make_args(estimator, "predict")[0]

    # check predict
    y_pred = getattr(estimator, "predict")(X)
    assert isinstance(y_pred, ACCEPTED_OUTPUT_TYPES)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isin(np.unique(y_pred), np.unique(y_train)))

    # check predict proba
    if hasattr(estimator, "predict_proba"):
        y_proba = getattr(estimator, "predict_proba")(X)
        assert isinstance(y_proba, ACCEPTED_OUTPUT_TYPES)
        assert y_proba.shape == (X.shape[0], N_CLASSES)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1)
        assert np.all(np.isin(np.unique(y_pred), np.unique(y_train)))


@pytest.mark.parametrize("Estimator", ALL_REGRESSORS)
def test_regressor_output(Estimator):
    estimator = _construct_instance(Estimator)
    X_train, y_train = _make_args(estimator, "fit")
    estimator.fit(X_train, y_train)

    X = _make_args(estimator, "predict")[0]

    # check predict
    y_pred = getattr(estimator, "predict")(X)
    assert isinstance(y_pred, ACCEPTED_OUTPUT_TYPES)
    assert y_pred.shape == (X.shape[0],)
    assert np.issubdtype(y_pred.dtype, np.floating)
