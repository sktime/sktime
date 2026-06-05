# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for X13ArimaSeats transformer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sktime.transformations.series.detrend._x13 import X13ArimaSeats


def test_X13ArimaSeats_raises_on_missing_binary():
    """Test that X13ArimaSeats raises FileNotFoundError when binary is missing."""
    X = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="ME"))
    transformer = X13ArimaSeats()

    with patch(
        "statsmodels.tsa.x13.x13_arima_analysis",
        side_effect=ValueError("x13as not found"),
    ):
        with pytest.raises(
            FileNotFoundError, match="X-13ARIMA-SEATS executable not found"
        ):
            transformer.fit(X)


def test_X13ArimaSeats_mocked_transform():
    """Test X13ArimaSeats fit and transform with a mocked backend."""
    X = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
        name="passengers",
    )

    mock_result = MagicMock()
    mock_result.seasadj = pd.Series([9.5, 19.5, 29.5], index=X.index, name="passengers")
    mock_result.trend = pd.Series([1.0, 2.0, 3.0], index=X.index)
    mock_result.seasonal = pd.Series([0.5, 0.5, 0.5], index=X.index)
    mock_result.irregular = pd.Series([0.5, 0.5, 0.5], index=X.index)

    with patch(
        "statsmodels.tsa.x13.x13_arima_analysis", return_value=mock_result
    ) as mock_x13:
        # Test return_components = False
        transformer = X13ArimaSeats(return_components=False)
        Xt = transformer.fit_transform(X)

        assert isinstance(Xt, pd.Series)
        assert Xt.name == "passengers"
        pd.testing.assert_series_equal(Xt, mock_result.seasadj)

        # Test inverse_transform (additive by default)
        X_inv = transformer.inverse_transform(Xt)
        pd.testing.assert_series_equal(X_inv, X)

        # Test return_components = True
        transformer_comp = X13ArimaSeats(return_components=True)
        Xt_comp = transformer_comp.fit_transform(X)

        assert isinstance(Xt_comp, pd.DataFrame)
        assert list(Xt_comp.columns) == ["seasadj", "trend", "seasonal", "irregular"]
        pd.testing.assert_series_equal(Xt_comp["seasadj"], mock_result.seasadj, check_names=False)

        # Test inverse_transform with components (additive)
        X_inv_comp = transformer_comp.inverse_transform(Xt_comp)
        assert isinstance(X_inv_comp, pd.DataFrame)
        pd.testing.assert_series_equal(X_inv_comp.iloc[:, 0], X)


def test_X13ArimaSeats_mocked_transform_multiplicative():
    """Test X13ArimaSeats multiplicative inverse transform with components."""
    X = pd.Series(
        [20.0, 40.0, 60.0],
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
        name="passengers",
    )

    mock_result_mult = MagicMock()
    mock_result_mult.seasadj = pd.Series([10.0, 20.0, 30.0], index=X.index)
    mock_result_mult.seasonal = pd.Series([2.0, 2.0, 2.0], index=X.index)
    mock_result_mult.trend = pd.Series([1.0, 2.0, 3.0], index=X.index)
    mock_result_mult.irregular = pd.Series([1.0, 1.0, 1.0], index=X.index)

    with patch(
        "statsmodels.tsa.x13.x13_arima_analysis", return_value=mock_result_mult
    ):
        transformer_mult = X13ArimaSeats(log=True, return_components=True)
        Xt_mult = transformer_mult.fit_transform(X)
        X_inv_mult = transformer_mult.inverse_transform(Xt_mult)
        assert isinstance(X_inv_mult, pd.DataFrame)
        pd.testing.assert_series_equal(
            X_inv_mult.iloc[:, 0], pd.Series([20.0, 40.0, 60.0], index=X.index, name="passengers")
        )
