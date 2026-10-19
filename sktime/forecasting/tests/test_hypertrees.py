import pandas as pd
import pytest

from sktime.forecasting.hypertrees._utils import _get_frequency


@pytest.mark.parametrize(
    ("legacy_frequency", "expected_frequency"),
    [
        ("M", "ME"),
        ("Q", "QE-DEC"),
        ("2M", "2ME"),
        ("Q-DEC", "QE-DEC"),
    ],
)
def test_hypertree_frequency_is_pandas_compatible(legacy_frequency, expected_frequency):
    """Deprecated pandas frequency aliases are normalized for Hyper-Trees."""
    index = type("IndexWithLegacyFrequency", (), {"freqstr": legacy_frequency})()

    assert _get_frequency(index) == expected_frequency


def test_hypertree_frequency_defaults_to_month_start():
    """Indexes without a frequency use the Hyper-Trees default."""
    index = pd.RangeIndex(3)

    assert _get_frequency(index) == "MS"
