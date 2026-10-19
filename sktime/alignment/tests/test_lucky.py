"""Tests for AlignerLuckyDtw."""

import numpy as np
import pandas as pd
import pytest

from sktime.alignment.lucky import AlignerLuckyDtw
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(AlignerLuckyDtw),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lucky_dtw_window_too_small_raises():
    """A window smaller than the length difference raises instead of hanging."""
    ts1 = pd.DataFrame({"a": np.arange(6.0)})
    ts2 = pd.DataFrame({"a": np.arange(2.0)})

    with pytest.raises(ValueError, match="warping window"):
        AlignerLuckyDtw(window=1).fit([ts1, ts2])


@pytest.mark.skipif(
    not run_test_for_class(AlignerLuckyDtw),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("window", [None, 4, 6])
def test_lucky_dtw_unequal_length_respects_window(window):
    """Alignment of series of unequal length ends at the last points, within window."""
    rng = np.random.default_rng(0)
    ts1 = pd.DataFrame({"a": rng.standard_normal(10)})
    ts2 = pd.DataFrame({"a": rng.standard_normal(6)})

    aligner = AlignerLuckyDtw(window=window).fit([ts1, ts2])
    align = aligner.get_alignment()

    assert align["ind0"].iloc[0] == 0 and align["ind1"].iloc[0] == 0
    assert align["ind0"].iloc[-1] == 9 and align["ind1"].iloc[-1] == 5
    # monotone, unit steps
    assert set(np.diff(align["ind0"])) <= {0, 1}
    assert set(np.diff(align["ind1"])) <= {0, 1}
    if window is not None:
        assert (align["ind0"] - align["ind1"]).abs().max() <= window
    assert aligner.get_distance() >= 0
