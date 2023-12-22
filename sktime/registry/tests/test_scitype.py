"""Tests for scitype typipng function."""

import pytest

from sktime.registry._scitype import scitype


@pytest.mark.parametrize("coerce_to_list", [True, False])
def test_scitype(coerce_to_list):
    """Test that the scitype function recovers the correct scitype(s)."""
    from sktime.forecasting.arima import ARIMA
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.transformations.series.exponent import ExponentTransformer

    # test that scitype works for classes with soft dependencies
    result_arima = scitype(ARIMA, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_arima, list)
        assert "forecaster" == result_arima[0]
    else:
        assert "forecaster" == result_arima

    # test that scitype works for instances
    result_naive = scitype(NaiveForecaster(), coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_naive, list)
        assert "forecaster" == result_naive[0]
    else:
        assert "forecaster" == result_naive

    # test transformer object
    result_transformer = scitype(ExponentTransformer, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_transformer, list)
        assert "transformer" == result_transformer[0]
    else:
        assert "transformer" == result_transformer
