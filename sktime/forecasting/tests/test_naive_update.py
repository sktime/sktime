"""Tests for NaiveForecaster._update method."""

import pytest
from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import temporal_train_test_split

@pytest.fixture
def airline_data():
    """Load airline dataset and split into train/test."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    return y_train, y_test


def test_update_no_warning(airline_data):
    """Test that _update does not trigger NotImplementedWarning."""
    import warnings
    y_train, y_test = airline_data
    f = NaiveForecaster(strategy="last", sp=12)
    f.fit(y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f.update(y_test.iloc[:3])


def test_update_cutoff_advances(airline_data):
    """Test that cutoff advances correctly after update."""
    y_train, y_test = airline_data
    f = NaiveForecaster(strategy="last", sp=12)
    f.fit(y_train)
    f.update(y_test.iloc[:3])
    assert f.cutoff[0] == y_test.index[2]


def test_update_window_length_grows(airline_data):
    """Test that window_length_ grows after update when window_length=None."""
    y_train, y_test = airline_data
    f = NaiveForecaster(strategy="last", sp=12)
    f.fit(y_train)
    wl_before = f.window_length_
    f.update(y_test.iloc[:3])
    assert f.window_length_ == wl_before + 3


def test_update_fixed_window_length_unchanged(airline_data):
    """Test that window_length_ stays fixed when window_length is set by user."""
    y_train, y_test = airline_data
    f = NaiveForecaster(strategy="mean", window_length=12, sp=12)
    f.fit(y_train)
    wl_before = f.window_length_
    f.update(y_test.iloc[:3])
    assert f.window_length_ == wl_before


def test_update_params_false_does_not_change_window(airline_data):
    """Test that update_params=False does not update internal state."""
    y_train, y_test = airline_data
    f = NaiveForecaster(strategy="last", sp=12)
    f.fit(y_train)
    wl_before = f.window_length_
    f.update(y_test.iloc[:3], update_params=False)
    assert f.window_length_ == wl_before


def test_update_predict_all_strategies(airline_data):
    """Test update followed by predict works for all strategies."""
    y_train, y_test = airline_data
    for strategy in ["last", "mean", "drift"]:
        f = NaiveForecaster(strategy=strategy)
        f.fit(y_train)
        f.update(y_test.iloc[:3])
        preds = f.predict(fh=[1, 2, 3])
        assert len(preds) == 3
        assert not preds.isna().any()