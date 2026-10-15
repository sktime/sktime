import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal

from sktime.alignment.lucky import AlignerLuckyDtw

def test_aligner_lucky_dtw_equal_length():
    """Test AlignerLuckyDtw with equal length series and specific window."""
    ts1 = pd.DataFrame({"dim_0": [1, 2, 3, 4]})
    ts2 = pd.DataFrame({"dim_0": [2, 3, 4, 5]})
    
    aligner = AlignerLuckyDtw(window=2)
    aligner.fit([ts1, ts2])
    
    alignment = aligner.get_alignment()
    dist = aligner.get_distance()
    
    expected_alignment = pd.DataFrame({
        "ind0": [0, 1, 2, 3, 3],
        "ind1": [0, 0, 1, 2, 3]
    })
    
    assert_frame_equal(alignment, expected_alignment)
    assert_almost_equal(dist, 2.0)


def test_aligner_lucky_dtw_unequal_length():
    """Test AlignerLuckyDtw with unequal length series."""
    ts1 = pd.DataFrame({"dim_0": [1, 2, 3, 4]})
    ts2 = pd.DataFrame({"dim_0": [1, 2, 3]})
    
    aligner = AlignerLuckyDtw(window=2)
    aligner.fit([ts1, ts2])
    
    alignment = aligner.get_alignment()
    dist = aligner.get_distance()
    
    expected_alignment = pd.DataFrame({
        "ind0": [0, 1, 2, 3],
        "ind1": [0, 1, 2, 2]
    })
    
    assert_frame_equal(alignment, expected_alignment)
    assert_almost_equal(dist, 1.0)


def test_aligner_lucky_dtw_no_window():
    """Test AlignerLuckyDtw fallback logic when window is None."""
    ts1 = pd.DataFrame({"dim_0": [1, 2, 3]})
    ts2 = pd.DataFrame({"dim_0": [1, 2, 3]})
    
    aligner = AlignerLuckyDtw(window=None)
    aligner.fit([ts1, ts2])
    
    # Internal window should have defaulted to max length (3)
    assert aligner.window is None
    
    alignment = aligner.get_alignment()
    dist = aligner.get_distance()
    
    expected_alignment = pd.DataFrame({
        "ind0": [0, 1, 2],
        "ind1": [0, 1, 2]
    })
    
    assert_frame_equal(alignment, expected_alignment)
    assert_almost_equal(dist, 0.0)
