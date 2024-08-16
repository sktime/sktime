"""Test functions for loose loaers."""

__author__ = ["fkiraly"]

__all__ = []


import pytest

from sktime.datatypes import check_raise
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_load_macroeconomic():
    """Test that load_macroeconomic runs."""
    from sktime.datasets import load_macroeconomic

    load_macroeconomic()


@pytest.mark.parametrize("special_dates", [True, False])
def test_load_stallion(special_dates):
    from sktime.datasets import load_stallion

    y, x = load_stallion(special_dates)
    check_raise(x, "pd_multiindex_hier")
    check_raise(y, "pd_multiindex_hier")
