#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import _BaseForecasterOptionalFHinFit
from sktime.forecasting.base import _BaseForecasterRequiredFHinFit
from sktime.forecasting.model_selection import RollingWindowSplit
from sktime.utils import all_estimators
from sktime.utils.testing import _construct_instance

# get all Forecasters
Forecasters = [e[1] for e in all_estimators(type_filter="forecaster")]

# divide Forecasters into groups
Forecasters_required_fh_in_fit = [f for f in Forecasters if issubclass(f, _BaseForecasterRequiredFHinFit)]
Forecasters_optional_fh_in_fit = [f for f in Forecasters if issubclass(f, _BaseForecasterOptionalFHinFit)]

########################################################################################################################
# test base api for setting/updating/getting fh
WINDOW_LENGTHS = [1, 3, 5]
STEP_LENGTHS = [1, 3, 5]
FHS = [1, 3, np.arange(1, 5)]
fh = FHS[0]

# testing data
n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


########################################################################################################################
# testing Forecasters which require fh during fitting
@pytest.mark.parametrize("Forecaster", Forecasters_required_fh_in_fit)
def test_no_fh_in_fit_req(Forecaster):
    f = _construct_instance(Forecaster)
    # fh required in fit, raises error if not passed
    with pytest.raises(ValueError):
        f.fit(y_train)


@pytest.mark.parametrize("Forecaster", Forecasters_required_fh_in_fit)
def test_fh_in_fit_req(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict()
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("Forecaster", Forecasters_required_fh_in_fit)
def test_same_fh_in_fit_and_predict_req(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("Forecaster", Forecasters_required_fh_in_fit)
def test_different_fh_in_fit_and_predict_req(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    np.testing.assert_array_equal(f.fh, fh)
    # updating fh during predict raises error as fitted model depends on fh seen in fit
    with pytest.raises(ValueError):
        f.predict(fh=fh + 1)


########################################################################################################################
# testing Forecasters which take fh either during fitting or predicting
@pytest.mark.parametrize("Forecaster", Forecasters_optional_fh_in_fit)
def test_no_fh_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train)
    # not passing fh to either fit or predict raises error
    with pytest.raises(ValueError):
        f.predict()


@pytest.mark.parametrize("Forecaster", Forecasters_optional_fh_in_fit)
def test_fh_in_fit_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    np.testing.assert_array_equal(f.fh, fh)
    f.predict()
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("Forecaster", Forecasters_optional_fh_in_fit)
def test_fh_in_predict_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("Forecaster", Forecasters_optional_fh_in_fit)
def test_same_fh_in_fit_and_predict_opt(Forecaster):
    f = _construct_instance(Forecaster)
    # passing the same fh to both fit and predict works
    f.fit(y_train, fh)
    f.predict(fh)
    np.testing.assert_array_equal(f.fh, fh)


@pytest.mark.parametrize("Forecaster", Forecasters_optional_fh_in_fit)
def test_different_fh_in_fit_and_predict_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    # passing different fh to predict than to fit works, but raises warning
    with pytest.warns(UserWarning):
        f.predict(fh + 1)
    np.testing.assert_array_equal(f.fh, fh + 1)


# check if warning is raised if inconsistent fh is passed
@pytest.mark.parametrize("Forecaster", Forecasters_optional_fh_in_fit)
@pytest.mark.parametrize("fh", FHS)
@pytest.mark.parametrize("window_length", WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", STEP_LENGTHS)
def test_update_predict_check_warning_for_inconsistent_fhs(Forecaster, fh, window_length, step_length):
    # check user warning if fh passed through cv is different from fh seen in fit
    cv = RollingWindowSplit(fh + 1, window_length=window_length, step_length=step_length)
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)

    # check for expected warning when updating fh via passed cv object
    with pytest.warns(UserWarning):
        f.update_predict(y_test, cv=cv)
