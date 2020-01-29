#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pytest
from sktime.forecasting.base import BaseForecasterOptionalFHinFit
from sktime.forecasting.base import BaseForecasterRequiredFHinFit

fh = np.array([1, 2])
y = np.random.normal(size=10)


# testing mixin classes for fh interface
# testing forecasters which require fh during fitting
class ForecasterReq(BaseForecasterRequiredFHinFit):

    def fit(self, y, fh=None, X=None):
        self._validate_fh(fh)
        self._is_fitted = True
        assert self.fh is not None

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        self._validate_fh(fh)
        assert self.fh is not None


def test_no_fh_in_fit_req():
    f = ForecasterReq()
    # fh required in fit, raises error if not passed
    with pytest.raises(ValueError):
        f.fit(y)


def test_fh_in_fit_req():
    f = ForecasterReq()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict()
    np.testing.assert_array_equal(f.fh, fh)


def test_same_fh_in_fit_and_predict_req():
    f = ForecasterReq()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


def test_different_fh_in_fit_and_predict_req():
    f = ForecasterReq()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    # updating fh during predict raises error as fitted model depends on fh seen in fit
    with pytest.raises(ValueError):
        f.predict(fh=fh + 1)


# testing forecasters which require fh either during fitting or predicting
class ForecasterOpt(BaseForecasterOptionalFHinFit):

    def fit(self, y, fh=None, X=None):
        self._validate_fh(fh)
        self._is_fitted = True

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        self._validate_fh(fh)
        assert self.fh is not None


def test_no_fh_opt():
    f = ForecasterOpt()
    f.fit(y)
    # not passing fh to either fit or predict raises error
    with pytest.raises(ValueError):
        f.predict()


def test_fh_in_fit_opt():
    f = ForecasterOpt()
    f.fit(y, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict()
    np.testing.assert_array_equal(f.fh, fh)


def test_fh_in_predict_opt():
    f = ForecasterOpt()
    f.fit(y)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


def test_same_fh_in_fit_and_predict_opt():
    f = ForecasterOpt()
    # passing the same fh to both fit and predict works
    f.fit(y, fh)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


def test_different_fh_in_fit_and_predict_opt():
    f = ForecasterOpt()
    f.fit(y, fh)
    # passing different fh to predict than to fit works, but raises warning
    with pytest.warns(UserWarning):
        f.predict(fh + 1)
    np.testing.assert_array_equal(f.fh, fh + 1)

