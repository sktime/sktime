# sktime/forecasting/tests/test_chronos2.py

from sktime.forecasting.chronos2 import Chronos2Forecaster


def test_import_chronos2():
    """Test that Chronos2Forecaster can be imported successfully."""
    _ = Chronos2Forecaster()
    assert True
