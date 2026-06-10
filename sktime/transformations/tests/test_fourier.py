"""Tests for the FourierFeatures transformer."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.fourier import FourierFeatures

Y = load_airline()
Y_datetime = deepcopy(Y)
Y_datetime.index = Y_datetime.index.to_timestamp(freq="M")


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fourier_list_length_mismatch():
    """Tests exception raised when sp_list & fourier_terms_list lengths don't match."""
    with pytest.raises(ValueError) as ex:
        FourierFeatures(sp_list=[365, 52], fourier_terms_list=[1])
        assert ex.value == (
            "In FourierFeatures the length of the sp_list needs to be equal "
            "to the length of fourier_terms_list."
        )


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fourier_k_larger_than_sp():
    """Tests exception raised when fourier_terms_list elements larger than sp_list."""
    with pytest.raises(ValueError) as ex:
        FourierFeatures(sp_list=[2], fourier_terms_list=[3])
        assert ex.value == (
            "In FourierFeatures the number of each element of fourier_terms_list"
            "needs to be lower from the corresponding element of the sp_list"
        )


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fourier_redundant_terms_dropped():
    """Tests redundant sp and k pairs are dropped if their equivalents exist."""
    transformer = FourierFeatures(sp_list=[12, 9, 3], fourier_terms_list=[4, 3, 1])
    transformer.fit(Y)
    assert transformer.sp_k_pairs_list_ == [
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (9, 1),
        (9, 2),
    ]


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_outputs():
    """Tests that we get the expected outputs."""
    y = Y.iloc[:3]
    y_transformed = FourierFeatures(
        sp_list=[12], fourier_terms_list=[2], keep_original_columns=True
    ).fit_transform(y)
    expected = (
        y.to_frame()
        .assign(sin_12_1=[np.sin(2 * np.pi * i / 12) for i in range(3)])
        .assign(cos_12_1=[np.cos(2 * np.pi * i / 12) for i in range(3)])
        .assign(sin_12_2=[np.sin(4 * np.pi * i / 12) for i in range(3)])
        .assign(cos_12_2=[np.cos(4 * np.pi * i / 12) for i in range(3)])
    )
    assert_frame_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_keep_original_columns_false():
    """Tests that we get the expected outputs when `keep_original_columns` is False."""
    y = Y.iloc[:3]
    y_transformed = FourierFeatures(
        sp_list=[12], fourier_terms_list=[2], keep_original_columns=False
    ).fit_transform(y)
    expected = (
        pd.DataFrame(index=y.index)
        .assign(sin_12_1=[np.sin(2 * np.pi * i / 12) for i in range(3)])
        .assign(cos_12_1=[np.cos(2 * np.pi * i / 12) for i in range(3)])
        .assign(sin_12_2=[np.sin(4 * np.pi * i / 12) for i in range(3)])
        .assign(cos_12_2=[np.cos(4 * np.pi * i / 12) for i in range(3)])
    )
    assert_frame_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_datetime_outputs():
    """Tests that we get expected outputs when the input has a pd.DatetimeIndex."""
    y = Y_datetime.iloc[:3]
    y_transformed = FourierFeatures(
        sp_list=[12], fourier_terms_list=[2], keep_original_columns=True
    ).fit_transform(y)
    expected = (
        y.to_frame()
        .assign(sin_12_1=[np.sin(2 * np.pi * i / 12) for i in range(3)])
        .assign(cos_12_1=[np.cos(2 * np.pi * i / 12) for i in range(3)])
        .assign(sin_12_2=[np.sin(4 * np.pi * i / 12) for i in range(3)])
        .assign(cos_12_2=[np.cos(4 * np.pi * i / 12) for i in range(3)])
    )
    assert_frame_equal(y_transformed, expected)
    assert_index_equal(y_transformed.index, y.index)


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_behaviour():
    """Tests that the transform method evaluates time steps passed based on X in fit."""
    transformer = FourierFeatures(
        sp_list=[12], fourier_terms_list=[2], keep_original_columns=True
    )
    # fit_transform on one part of the dataset
    y_tr_1 = transformer.fit_transform(Y.iloc[:100])
    # transform the rest
    y_tr_2 = transformer.transform(Y.iloc[100:])
    # fit_transform the entire dataset
    y_tr_complete = transformer.fit_transform(Y)
    assert_frame_equal(pd.concat([y_tr_1, y_tr_2]), y_tr_complete)
