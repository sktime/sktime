import numpy as np
import pandas as pd
import pytest

from sktime.alignment.naive import AlignerNaive

def test_aligner_naive_start():
    # Test 'start' strategy: pads at the end
    X = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4, 5]})]
    aligner = AlignerNaive(strategy="start")
    aligner.fit(X)
    align = aligner.get_alignment()
    
    # max length is 3. 
    # ind0 should be [0, 1, np.nan] (pad at end)
    # ind1 should be [0, 1, 2]
    np.testing.assert_array_equal(align["ind0"][:2].values, [0, 1])
    assert pd.isna(align["ind0"].iloc[2])
    np.testing.assert_array_equal(align["ind1"].values, [0, 1, 2])

def test_aligner_naive_end():
    # Test 'end' strategy: pads at the start
    X = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4, 5]})]
    aligner = AlignerNaive(strategy="end")
    aligner.fit(X)
    align = aligner.get_alignment()
    
    # max length is 3. 
    # ind0 should be [np.nan, 0, 1] (pad at start)
    # ind1 should be [0, 1, 2]
    assert pd.isna(align["ind0"].iloc[0])
    np.testing.assert_array_equal(align["ind0"].iloc[1:].values, [0, 1])
    np.testing.assert_array_equal(align["ind1"].values, [0, 1, 2])

def test_aligner_naive_start_end():
    # Test 'start-end' strategy: stretch linearly
    X = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4, 5, 6]})]
    aligner = AlignerNaive(strategy="start-end")
    aligner.fit(X)
    align = aligner.get_alignment()
    
    # max length is 4.
    # length of X[0] is 2, length of X[1] is 4
    # ind0: linspace(0, 1, 4) -> [0, 0.33, 0.66, 1] -> round -> [0, 0, 1, 1]
    # ind1: linspace(0, 3, 4) -> [0, 1, 2, 3]
    np.testing.assert_array_equal(align["ind0"].values, [0, 0, 1, 1])
    np.testing.assert_array_equal(align["ind1"].values, [0, 1, 2, 3])

def test_aligner_naive_invalid_strategy():
    # Test invalid strategy
    aligner = AlignerNaive(strategy="invalid")
    X = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
    with pytest.raises(ValueError, match="strategy must be one of"):
        aligner.fit(X)
