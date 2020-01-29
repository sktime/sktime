#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pytest

from sktime.forecasting.base import BaseForecasterRequiredFHinFit
from sktime.forecasting.dummy import DummyForecaster


# testing forecasters which require fh during fitting
# to be replaced when we have actual forecasters which require fh in fitting
class ForecasterReq(BaseForecasterRequiredFHinFit):

    def fit(self, y, fh=None, X=None):
        self._set_fh(fh)
        self._is_fitted = True
        assert self.fh is not None

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        self._set_fh(fh)
        assert self.fh is not None


FORECASTERS_REQUIRED_FH = [ForecasterReq]
FORECASTERS_OPTIONAL_FH = [DummyForecaster]

fh = np.array([1, 2])
y = np.random.normal(size=10)


@pytest.mark.parametrize("forecaster", FORECASTERS_REQUIRED_FH)
def test_no_fh_in_fit_req(forecaster):
    f = forecaster()
    # fh required in fit, raises error if not passed
    with pytest.raises(ValueError):
        f.fit(y)


@pytest.mark.parametrize("forecaster", FORECASTERS_REQUIRED_FH)
def test_fh_in_fit_req(forecaster):
    f = forecaster()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict()
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("forecaster", FORECASTERS_REQUIRED_FH)
def test_same_fh_in_fit_and_predict_req(forecaster):
    f = forecaster()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("forecaster", FORECASTERS_REQUIRED_FH)
def test_different_fh_in_fit_and_predict_req(forecaster):
    f = forecaster()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    # updating fh during predict raises error as fitted model depends on fh seen in fit
    with pytest.raises(ValueError):
        f.predict(fh=fh + 1)


# testing forecasters which require fh either during fitting or predicting
@pytest.mark.parametrize("forecaster", FORECASTERS_OPTIONAL_FH)
def test_no_fh_opt(forecaster):
    f = forecaster()
    f.fit(y)
    # not passing fh to either fit or predict raises error
    with pytest.raises(ValueError):
        f.predict()


@pytest.mark.parametrize("forecaster", FORECASTERS_OPTIONAL_FH)
def test_fh_in_fit_opt(forecaster):
    f = forecaster()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict()
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("forecaster", FORECASTERS_OPTIONAL_FH)
def test_fh_in_predict_opt(forecaster):
    f = forecaster()
    f.fit(y)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("forecaster", FORECASTERS_OPTIONAL_FH)
def test_same_fh_in_fit_and_predict_opt(forecaster):
    f = forecaster()
    # passing the same fh to both fit and predict works
    f.fit(y, fh)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("forecaster", FORECASTERS_OPTIONAL_FH)
def test_different_fh_in_fit_and_predict_opt(forecaster):
    f = forecaster()
    f.fit(y, fh)
    # passing different fh to predict than to fit works, but raises warning
    with pytest.warns(UserWarning):
        f.predict(fh + 1)
    np.testing.assert_array_equal(f.fh, fh + 1)
