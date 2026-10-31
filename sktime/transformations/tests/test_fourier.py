"""Tests for the FourierFeatures transformer."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.fourier import FourierFeatures

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


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_month_start_outputs():
    """Tests expected outputs for a month start pd.DatetimeIndex, see pandas 3."""
    y = Y.iloc[:3].copy()
    y.index = y.index.to_timestamp()
    assert isinstance(y.index.freq, pd.offsets.MonthBegin)
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
@pytest.mark.parametrize(
    "freq, sp",
    [("30min", 48), ("15min", 96), ("2h", 12), ("4h", 6), ("h", 24)],
)
def test_integer_sp_counts_index_steps(freq, sp):
    """Integer sp is in index steps, so it should agree with the offset "D", see #5933.

    For a multiplied frequency such as ``30min``, the integer ordinals of the
    period index advance by the multiplier per step, which used to be applied
    directly, so that ``sp=48`` was interpreted as 48 minutes rather than 48
    half-hourly steps.
    """
    n = 3 * sp
    y = pd.DataFrame(
        {"x": np.arange(n, dtype=float)},
        index=pd.date_range("2000-01-01", periods=n, freq=freq),
    )
    transformer = FourierFeatures(sp_list=[sp, "D"], fourier_terms_list=[1, 1])
    y_transformed = transformer.fit_transform(y)

    # integer sp should be equal to the daily offset, since sp steps make up a day
    np.testing.assert_allclose(
        y_transformed[f"sin_{sp}_1"].to_numpy(),
        y_transformed["sin_D_1"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        y_transformed[f"cos_{sp}_1"].to_numpy(),
        y_transformed["cos_D_1"].to_numpy(),
        atol=1e-12,
    )

    # and equal to the explicit definition in time steps since the start of fit
    t = np.arange(n)
    np.testing.assert_allclose(
        y_transformed[f"sin_{sp}_1"].to_numpy(), np.sin(2 * np.pi * t / sp)
    )
    np.testing.assert_allclose(
        y_transformed[f"cos_{sp}_1"].to_numpy(), np.cos(2 * np.pi * t / sp)
    )


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_integer_sp_multiplied_freq_transform_continues_from_fit():
    """Integer sp with multiplied frequency, transform on data after the fit data."""
    n_fit, n_transform = 100, 50
    idx = pd.date_range("2000-01-01", periods=n_fit + n_transform, freq="30min")
    y = pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)

    transformer = FourierFeatures(sp_list=[48], fourier_terms_list=[2])
    transformer.fit(y.iloc[:n_fit])
    y_transformed = transformer.transform(y.iloc[n_fit:])

    t = np.arange(n_fit, n_fit + n_transform)
    expected = pd.DataFrame(
        {
            "sin_48_1": np.sin(2 * np.pi * t / 48),
            "cos_48_1": np.cos(2 * np.pi * t / 48),
            "sin_48_2": np.sin(4 * np.pi * t / 48),
            "cos_48_2": np.cos(4 * np.pi * t / 48),
        },
        index=idx[n_fit:],
    )
    assert_frame_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class(FourierFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_integer_sp_period_index_multiplied_freq():
    """Integer sp counts index steps for a pd.PeriodIndex with multiplied freq."""
    n = 60
    y = pd.Series(
        np.arange(n, dtype=float),
        index=pd.period_range("2000-01-01", periods=n, freq="2h"),
    )
    y_transformed = FourierFeatures(sp_list=[12], fourier_terms_list=[1]).fit_transform(
        y
    )

    t = np.arange(n)
    np.testing.assert_allclose(
        y_transformed["sin_12_1"].to_numpy(), np.sin(2 * np.pi * t / 12)
    )
    np.testing.assert_allclose(
        y_transformed["cos_12_1"].to_numpy(), np.cos(2 * np.pi * t / 12)
    )
