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


@pytest.mark.parametrize("force_single_scitype", [True, False])
@pytest.mark.parametrize("coerce_to_list", [True, False])
def test_scitype_generic(force_single_scitype, coerce_to_list):
    """Test that the scitype function recovers the correct scitype(s)."""
    from sktime.base import BaseObject

    class _DummyClass(BaseObject):
        _tags = {"object_type": ["foo", "bar"]}

    scitype_inferred = scitype(
        _DummyClass(),
        force_single_scitype=force_single_scitype,
        coerce_to_list=coerce_to_list,
    )

    if force_single_scitype and coerce_to_list:
        expected = ["foo"]
    if not force_single_scitype and coerce_to_list:
        expected = ["foo", "bar"]
    if not coerce_to_list:
        expected = "foo"

    assert scitype_inferred == expected

    class _DummyClass2(BaseObject):
        _tags = {"object_type": "foo"}

    scitype_inferred = scitype(
        _DummyClass2(),
        force_single_scitype=force_single_scitype,
        coerce_to_list=coerce_to_list,
    )

    if coerce_to_list:
        expected = ["foo"]
    if not coerce_to_list:
        expected = "foo"

    assert scitype_inferred == expected

    class _DummyClass3(BaseObject):
        _tags = {"object_type": ["foo"]}

    scitype_inferred = scitype(
        _DummyClass3(),
        force_single_scitype=force_single_scitype,
        coerce_to_list=coerce_to_list,
    )

    if coerce_to_list:
        expected = ["foo"]
    if not coerce_to_list:
        expected = "foo"

    assert scitype_inferred == expected
