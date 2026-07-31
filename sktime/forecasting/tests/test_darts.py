"""Test for Darts Models."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

import importlib
import re

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datasets import load_longley
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


def _darts_to_series(obj):
    """Convert Darts object to pandas Series."""
    # darts changed the name of the method converting "TimeSeries" to "pandas.Series"
    # in version 0.35, so we need to check the version
    # also, darts is distributed as "darts" and "u8darts", so we need to check both
    darts_ge_035 = [
        _check_soft_dependencies("darts>=0.35", severity="none")
        or _check_soft_dependencies("u8darts>=0.35", severity="none")
    ]
    if darts_ge_035:
        to_ser_name = "to_series"
    else:
        to_ser_name = "pd_series"
    return getattr(obj, to_ser_name)()


def _load_weather_pressure_head(n_timepoints=100):
    """Load the first ``n_timepoints`` of WeatherDataset atmospheric pressure.

    ``WeatherDataset().load()`` parses the entire CSV. A truncated download can
    fail mid-file at a fixed offset (CI has seen ``time data "10"`` at row
    23251 — the start of ``10.06.2020 ...``). This helper still uses darts'
    download + MD5 check, but only date-parses the leading rows we need. If a
    previous run cached a truncated file, the hash check fails and we re-download.
    """
    from darts import TimeSeries
    from darts.datasets import WeatherDataset
    from darts.datasets.dataset_loaders import DatasetLoadingException

    dataset = WeatherDataset()
    # trigger download and integrity check without parsing the full CSV
    if not dataset._is_already_downloaded():
        dataset._download_dataset()
    try:
        dataset._check_dataset_integrity_or_raise()
    except DatasetLoadingException:
        # common CI failure mode: partial download left on disk
        path = dataset._get_path_dataset()
        path.unlink(missing_ok=True)
        dataset._download_dataset()
        dataset._check_dataset_integrity_or_raise()

    path = dataset._get_path_dataset()
    df = pd.read_csv(path, nrows=n_timepoints)
    df["Date Time"] = pd.to_datetime(
        df["Date Time"], format="%d.%m.%Y %H:%M:%S", errors="raise"
    )
    y = df.set_index("Date Time")["p (mbar)"]
    y.name = "p (mbar)"
    return TimeSeries.from_series(y), y


@pytest.mark.parametrize("model", [DartsXGBModel, DartsLinearRegressionModel])
@pytest.mark.skipif(
    not run_test_for_class([DartsXGBModel, DartsLinearRegressionModel]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_regression_models_with_weather_dataset(model):
    """Test with weather dataset."""
    kwargs = model_kwargs.get(model, {})
    model_to_import = import_mappings.get(model)
    # Create and fit the model
    imported_model = getattr(importlib.import_module("darts.models"), model_to_import)
    darts_model = imported_model(lags=12, output_chunk_length=6, **kwargs)
    target, target_df = _load_weather_pressure_head()

    darts_model.fit(target)
    # Make a prediction for the next 6 time steps
    darts_pred = _darts_to_series(darts_model.predict(6))
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
    from sklearn.ensemble import RandomForestRegressor

    model_to_import = import_mappings.get(model)
    # Create and fit the model
    imported_model = getattr(importlib.import_module("darts.models"), model_to_import)
    darts_model = imported_model(
        lags=12, output_chunk_length=6, model=RandomForestRegressor(random_state=0)
    )
    target, target_df = _load_weather_pressure_head()

    darts_model.fit(target)

    # Make a prediction for the next 6 time steps
    darts_pred = _darts_to_series(darts_model.predict(6))
    assert isinstance(target_df, pd.Series)
    sktime_model = model(
        lags=12,
        output_chunk_length=6,
        model=RandomForestRegressor(random_state=0),
    )
    sktime_model.fit(target_df)
    fh = list(range(1, 7))
    pred_sktime = sktime_model.predict(fh)
    assert isinstance(pred_sktime, pd.Series)

    np.testing.assert_allclose(pred_sktime.to_numpy(), darts_pred.to_numpy(), rtol=1e-4)
