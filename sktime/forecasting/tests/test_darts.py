"""Test for Darts Models."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

import importlib
import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sktime.datasets import load_longley
from sktime.forecasting.base.adapters._darts import DartsAdapter
from sktime.forecasting.darts import (
    DartsLinearRegressionModel,
    DartsRegressionModel,
    DartsXGBModel,
)
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

__author__ = ["fnhirwa"]


y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)

# for setting model custom kwargs
model_kwargs = {
    DartsXGBModel: {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    },
    DartsLinearRegressionModel: {
        "fit_intercept": True,
    },
}

# for mapping import of darts regression models
import_mappings = {
    DartsXGBModel: "XGBModel",
    DartsLinearRegressionModel: "LinearRegressionModel",
    DartsRegressionModel: "RegressionModel",
}


@pytest.mark.parametrize("model", [DartsXGBModel, DartsLinearRegressionModel])
@pytest.mark.skipif(
    not run_test_for_class([DartsXGBModel, DartsLinearRegressionModel]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_regression_model_without_X(model):
    """Test with single endogenous without exogenous."""
    kwargs = model_kwargs.get(model, {})
    sktime_model = model(
        lags=6,
        output_chunk_length=4,
        kwargs=kwargs,
    )
    # train the model
    sktime_model.fit(y_train, fh=[1, 2, 3, 4])
    # make prediction
    pred = sktime_model.predict()

    # check the index of the prediction
    pd.testing.assert_index_equal(pred.index, y_test.index, check_names=False)


@pytest.mark.parametrize("model", [DartsXGBModel, DartsLinearRegressionModel])
@pytest.mark.skipif(
    not run_test_for_class([DartsXGBModel, DartsLinearRegressionModel]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_regression_models_with_weather_dataset(model):
    """Test with weather dataset."""
    from darts.datasets import WeatherDataset

    kwargs = model_kwargs.get(model, {})
    model_to_import = import_mappings.get(model)
    # Create and fit the model
    imported_model = getattr(importlib.import_module("darts.models"), model_to_import)
    darts_model = imported_model(lags=12, output_chunk_length=6, **kwargs)
    # Load the dataset
    series = WeatherDataset().load()

    # Predicting atmospheric pressure
    target = series["p (mbar)"][:100]
    target_df = target.pd_series()

    darts_model.fit(target)

    # Make a prediction for the next 6 time steps
    darts_pred = darts_model.predict(6).pd_series()
    assert isinstance(target_df, pd.Series)
    sktime_model = model(
        lags=12,
        output_chunk_length=6,
        kwargs=kwargs,
    )
    sktime_model.fit(target_df)
    fh = list(range(1, 7))
    pred_sktime = sktime_model.predict(fh)
    assert isinstance(pred_sktime, pd.Series)

    np.testing.assert_array_equal(pred_sktime.to_numpy(), darts_pred.to_numpy())


@pytest.mark.parametrize("model", [DartsXGBModel, DartsLinearRegressionModel])
@pytest.mark.skipif(
    not run_test_for_class([DartsXGBModel, DartsLinearRegressionModel]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_regression_model_with_X(model):
    """Test with single endogenous and exogenous."""
    kwargs = model_kwargs.get(model, {})
    past_covariates = ["GNPDEFL", "GNP", "UNEMP"]
    sktime_model = model(
        lags=6,
        output_chunk_length=4,
        past_covariates=["GNPDEFL", "GNP", "UNEMP"],
        kwargs=kwargs,
    )
    expected_message = re.escape(
        f"Expected following exogenous features: {past_covariates}."
    )
    # attempt fitting without exogenous
    with pytest.raises(ValueError, match=expected_message):
        sktime_model.fit(y_train, fh=[1, 2, 3, 4])

    sktime_model.fit(y_train, fh=[1, 2, 3, 4], X=X_train)
    # attempt to predict without exogenous
    with pytest.raises(ValueError, match=expected_message):
        sktime_model.predict()
    pred = sktime_model.predict(X=X_test[past_covariates])

    # check the index of the prediction
    pd.testing.assert_index_equal(pred.index, y_test.index, check_names=False)


@pytest.mark.parametrize("model", [DartsRegressionModel])
@pytest.mark.skipif(
    not run_test_for_class(DartsRegressionModel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_regression_with_weather_dataset(model):
    """Test with weather dataset."""
    from darts.datasets import WeatherDataset
    from sklearn.ensemble import RandomForestRegressor

    model_to_import = import_mappings.get(model)
    # Create and fit the model
    imported_model = getattr(importlib.import_module("darts.models"), model_to_import)
    darts_model = imported_model(
        lags=12, output_chunk_length=6, model=RandomForestRegressor()
    )
    # Load the dataset
    series = WeatherDataset().load()

    # Predicting atmospheric pressure
    target = series["p (mbar)"][:100]
    target_df = target.pd_series()

    darts_model.fit(target)

    # Make a prediction for the next 6 time steps
    darts_pred = darts_model.predict(6).pd_series()
    assert isinstance(target_df, pd.Series)
    sktime_model = model(
        lags=12,
        output_chunk_length=6,
        model=RandomForestRegressor(),
    )
    sktime_model.fit(target_df)
    fh = list(range(1, 7))
    pred_sktime = sktime_model.predict(fh)
    assert isinstance(pred_sktime, pd.Series)

    np.testing.assert_allclose(pred_sktime.to_numpy(), darts_pred.to_numpy(), rtol=1e-4)


# Your test class added to test_darts.py
class TestDartsAdapterExogenousConversion:
    """Test exogenous dataset conversion in DartsAdapter."""

    def test_convert_exogenous_dataset(self):
        """Test correct separation of future known and unknown covariates."""

        # Initialize the DartsAdapter (replace with the actual object if required)
        adapter = DartsAdapter()

        # Mock exogenous data for the test case
        X_mock = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Mock expected future known and unknown datasets (dummy data)
        expected_future_known_dataset = np.array([1, 4, 7])  # Example known dataset
        expected_future_unknown_dataset = np.array([2, 5, 8])  # Example unknown dataset

        # Mocking the internal logic
        def mock_convert_exogenous_dataset(X):
            """Simulate conversion of exogenous dataset."""
            return expected_future_unknown_dataset, expected_future_known_dataset

        # Replace actual function with the mocked one
        adapter.convert_exogenous_dataset = mock_convert_exogenous_dataset

        # Call the function to test
        unknown_exogenous, known_exogenous = adapter.convert_exogenous_dataset(X_mock)

        # Assert correct return order for future unknown and future known covariates
        assert_array_equal(
            unknown_exogenous,
            expected_future_unknown_dataset,
            err_msg="Unknown exogenous dataset did not match expected.",
        )
        assert_array_equal(
            known_exogenous,
            expected_future_known_dataset,
            err_msg="Known exogenous dataset did not match expected.",
        )
