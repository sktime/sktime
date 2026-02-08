"""Unit tests for DegreeDayFeatures (sktime transformer).

These tests validate:
- Correctness on hand-checkable examples.
- Invariants (non-negativity, mutually exclusive HDD/CDD positivity).
- Input validation (missing columns, non-numeric values).
- Handling of inverted min/max temperatures (strict vs auto-swap).
"""

# Third-party imports.
import pandas as pd
import pytest

# Local imports (within sktime).
from sktime.transformations.series.degree_day import DegreeDayFeatures
from sktime.tests.test_switch import run_test_for_class

###############################################################################
# Test helpers.
###############################################################################


def _df_high_low(high, low, idx=None) -> pd.DataFrame:
    """Create a DataFrame with standard 'high'/'low' columns."""
    if idx is not None:
        return pd.DataFrame({"high": high, "low": low}, index=idx)
    return pd.DataFrame({"high": high, "low": low})


###############################################################################
# Correctness tests.
###############################################################################


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_basic_values():
    """
    Verify exact outputs on a small hand-checkable dataset.

    base_temp=65, tmean = (high+low)/2
    - tmean: [50, 65, 80]
    - HDD:   [15, 0, 0]
    - CDD:   [0, 0, 15]
    """
    X = _df_high_low(
        high=[60, 70, 90],
        low=[40, 60, 70],
        idx=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    )

    tx = DegreeDayFeatures(
        base_temp=65.0, return_tmean=True, keep_original_columns=False
    )
    out = tx.fit_transform(X)

    assert list(out.columns) == ["tmean", "hdd", "cdd"]

    assert out.loc["2025-01-01", "tmean"] == 50.0
    assert out.loc["2025-01-01", "hdd"] == 15.0
    assert out.loc["2025-01-01", "cdd"] == 0.0

    assert out.loc["2025-01-02", "tmean"] == 65.0
    assert out.loc["2025-01-02", "hdd"] == 0.0
    assert out.loc["2025-01-02", "cdd"] == 0.0

    assert out.loc["2025-01-03", "tmean"] == 80.0
    assert out.loc["2025-01-03", "hdd"] == 0.0
    assert out.loc["2025-01-03", "cdd"] == 15.0


###############################################################################
# Invariant tests.
###############################################################################


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_invariants():
    """Basic invariants should always hold for valid inputs."""
    X = _df_high_low(
        high=[80, 30],
        low=[70, 10],
        idx=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )

    tx = DegreeDayFeatures(
        base_temp=65.0, return_tmean=True, keep_original_columns=False
    )
    out = tx.fit_transform(X)

    # Non-negativity.
    assert (out["hdd"] >= 0).all()
    assert (out["cdd"] >= 0).all()

    # A day cannot be both above and below base temp under standard definition.
    assert ((out["hdd"] > 0) & (out["cdd"] > 0)).sum() == 0

    # tmean should be between low and high.
    assert (out["tmean"] >= X["low"]).all()
    assert (out["tmean"] <= X["high"]).all()

    # Index must be preserved.
    assert out.index.equals(X.index)


###############################################################################
# Input validation tests.
###############################################################################


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_missing_column_raises():
    """Missing required columns should raise when explicit columns are requested."""
    X = pd.DataFrame({"high": [70, 80]})
    tx = DegreeDayFeatures(tmax_col="high", tmin_col="low")  # explicit mode

    with pytest.raises(ValueError):
        tx.fit_transform(X)


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_non_numeric_raises():
    """Non-numeric temperature values should raise."""
    X = pd.DataFrame({"high": [70, "hot"], "low": [50, 40]})
    tx = DegreeDayFeatures()

    with pytest.raises(ValueError):
        tx.fit_transform(X)


###############################################################################
# Inverted min/max behavior tests.
###############################################################################


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_inverted_min_max_strict_raises():
    """If strict=True, rows where low > high should raise."""
    X = _df_high_low(high=[60], low=[70])
    tx = DegreeDayFeatures(strict=True)

    with pytest.raises(ValueError):
        tx.fit_transform(X)


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_inverted_min_max_swaps_when_not_strict():
    """If strict=False, rows where low > high should be auto-swapped."""
    X = _df_high_low(high=[60], low=[70])
    tx = DegreeDayFeatures(
        strict=False, base_temp=65.0, return_tmean=True, keep_original_columns=False
    )
    out = tx.fit_transform(X)

    # swapped => low=60, high=70 => tmean=65 => hdd=0 cdd=0
    assert out.loc[0, "tmean"] == 65.0
    assert out.loc[0, "hdd"] == 0.0
    assert out.loc[0, "cdd"] == 0.0


@pytest.mark.skipif(
    not run_test_for_class(DegreeDayFeatures),
    reason="run_test_for_class marks if the test should be run for this class",
)
def test_keep_original_columns_appends_features():
    """If keep_original_columns=True, features should be appended to X."""
    X = _df_high_low(high=[80], low=[60])
    tx = DegreeDayFeatures(keep_original_columns=True, return_tmean=True)
    out = tx.fit_transform(X)

    # original columns still present
    assert "high" in out.columns
    assert "low" in out.columns

    # features appended
    assert "tmean" in out.columns
    assert "hdd" in out.columns
    assert "cdd" in out.columns
