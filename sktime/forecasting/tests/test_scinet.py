"""Tests for SCINetForecaster input validation."""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datasets import load_airline


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency torch not available",
)
def test_scinet_invalid_seq_len():
    """Test that SCINetForecaster raises ValueError when seq_len > len(y)."""
    from sktime.forecasting.scinet import SCINetForecaster

    y = load_airline()[:15]  # Series of length 15
    forecaster = SCINetForecaster(seq_len=16, num_levels=2, pred_len=4)

    with pytest.raises(ValueError, match="<= the length of the training data"):
        forecaster.fit(y, fh=[1, 2])


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency torch not available",
)
def test_scinet_indivisible_seq_len():
    """Test that SCINetForecaster raises ValueError when seq_len not divisible."""
    from sktime.forecasting.scinet import SCINetForecaster

    with pytest.raises(ValueError, match=r"divisible by 2\^num_levels"):
        # seq_len=14 is not divisible by 2^2 = 4
        SCINetForecaster(seq_len=14, num_levels=2, pred_len=4)
