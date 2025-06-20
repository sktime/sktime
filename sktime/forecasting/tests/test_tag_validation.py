"""Tests for missing values capability validation in forecasters."""

import numpy as np
import pandas as pd
import pytest

from sktime.registry import all_estimators


def _create_missing_value_test_data():
    """Create test data with missing values."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")

    values = np.random.randn(20)
    values[5] = np.nan
    values[10] = np.nan
    values[15] = np.nan
    y_missing = pd.Series(values, index=dates)

    X_missing = pd.DataFrame(
        {"x1": np.random.randn(20), "x2": np.random.randn(20)}, index=dates
    )
    X_missing.iloc[7, 0] = np.nan
    X_missing.iloc[12, 1] = np.nan

    return y_missing, X_missing


def test_missing_values_capability_accuracy():
    """Test forecasters with capability:missing_values=True can handle NaNs."""
    y_missing, X_missing = _create_missing_value_test_data()

    forecasters = all_estimators(
        estimator_types="forecaster",
        return_names=False,
        exclude_estimators=["ElasticEnsemble"],
    )

    skip_forecasters = {
        "Prophet",
        "NeuralForecastRNN",
        "PytorchForecaster",
        "TensorFlowForecaster",
        "AutoARIMA",
        "ARIMA",
        "DynamicFactor",
        "UnobservedComponents",
    }

    for forecaster_class in forecasters:
        if forecaster_class.__name__ in skip_forecasters:
            continue

        try:
            forecaster = forecaster_class.create_test_instance()

            can_handle_missing = forecaster.get_tag("capability:missing_values", False)
            requires_fh_in_fit = forecaster.get_tag("requires-fh-in-fit", False)

            if can_handle_missing:
                try:
                    if requires_fh_in_fit:
                        forecaster.fit(y_missing, fh=[1, 2, 3])
                    else:
                        forecaster.fit(y_missing)
                    forecaster.predict(fh=[1, 2, 3])
                except Exception as e:
                    pytest.fail(
                        f"{forecaster_class.__name__} claims "
                        f"capability:missing_values=True but failed: {str(e)}"
                    )
            else:
                with pytest.raises((ValueError, NotImplementedError)) as exc_info:
                    if requires_fh_in_fit:
                        forecaster.fit(y_missing, fh=[1, 2, 3])
                    else:
                        forecaster.fit(y_missing)

                error_msg = str(exc_info.value).lower()
                assert any(word in error_msg for word in ["nan", "missing"]), (
                    f"{forecaster_class.__name__} should raise error about "
                    f"missing values, got: {str(exc_info.value)}"
                )

        except Exception as e:
            if any(
                skip_word in str(e).lower()
                for skip_word in ["import", "dependency", "optional", "install"]
            ):
                continue
            else:
                raise
