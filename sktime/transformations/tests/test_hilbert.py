# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for HilbertTransformer."""

__author__ = ["ved197338"]

import numpy as np
import pandas as pd
import pytest

from sktime.transformations.hilbert import HilbertTransformer


def test_envelope_tracks_amplitude_modulation():
    """Envelope of an AM signal should approximate the modulating waveform."""
    t = np.linspace(0, 2 * np.pi, 200)
    true_envelope = 1.0 + 0.5 * np.sin(t)
    carrier = np.sin(20 * t)
    signal = true_envelope * carrier

    transformer = HilbertTransformer(output_type="envelope")
    estimated = transformer.fit_transform(pd.Series(signal, name="am"))

    assert isinstance(estimated, pd.Series)
    assert len(estimated) == len(signal)

    # check the middle section (edges have boundary artifacts)
    mae = np.mean(np.abs(estimated.to_numpy()[20:-20] - true_envelope[20:-20]))
    assert mae < 0.15, f"envelope MAE too high: {mae:.3f}"


def test_all_output_gives_three_columns():
    """output_type='all' should produce envelope, phase, frequency columns."""
    df = pd.DataFrame({"x": np.sin(np.linspace(0, 10, 100))})
    result = HilbertTransformer(output_type="all", fs=10.0).fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x_envelope", "x_phase", "x_frequency"}
    assert len(result) == 100


def test_rejects_invalid_output_type():
    """Passing a bogus output_type should raise ValueError."""
    with pytest.raises(ValueError, match="output_type"):
        t = HilbertTransformer(output_type="not_a_real_option")
        t.fit_transform(pd.Series([1.0, 2.0, 3.0]))


def test_hilbert_rejects_N_smaller_than_signal_length():
    """N < len(X) should raise ValueError to prevent length mismatch."""
    series = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="must be greater than or equal"):
        HilbertTransformer(N=5).fit_transform(series)


def test_hilbert_valid_N_larger_than_signal_length():
    """N >= len(X) should pad FFT and return original length output."""
    series = pd.Series(np.arange(10, dtype=float))
    result = HilbertTransformer(N=16).fit_transform(series)
    assert len(result) == 10


def test_hilbert_numerical_sine_quadrature():
    """For sin(wt), quadrature component should be -cos(wt)."""
    t = np.linspace(0, 4 * np.pi, 200)
    series = pd.Series(np.sin(t))
    quad = HilbertTransformer(output_type="quadrature").fit_transform(series)
    expected = -np.cos(t)
    mae = np.mean(np.abs(quad.to_numpy()[20:-20] - expected[20:-20]))
    assert mae < 0.05

def test_hilbert_length_1_frequency_and_all():
    """Length-1 inputs should return 0.0 frequency without ValueError."""
    s = pd.Series([2.5], name="val")

    t_freq = HilbertTransformer(output_type="frequency")
    res_freq = t_freq.fit_transform(s)
    assert isinstance(res_freq, pd.Series)
    assert len(res_freq) == 1
    assert res_freq.iloc[0] == 0.0

    t_all = HilbertTransformer(output_type="all")
    res_all = t_all.fit_transform(s)
    assert isinstance(res_all, pd.DataFrame)
    assert res_all.shape == (1, 3)
    assert res_all["val_frequency"].iloc[0] == 0.0
    assert res_all["val_envelope"].iloc[0] == 2.5


def test_hilbert_empty_input():
    """Empty inputs should return empty Series/DataFrame without error."""
    s = pd.Series([], dtype=float, name="val")

    for ot in ["envelope", "phase", "quadrature", "frequency"]:
        res = HilbertTransformer(output_type=ot).fit_transform(s)
        assert isinstance(res, pd.Series)
        assert len(res) == 0

    res_all = HilbertTransformer(output_type="all").fit_transform(s)
    assert isinstance(res_all, pd.DataFrame)
    assert len(res_all) == 0
    assert set(res_all.columns) == {"val_envelope", "val_phase", "val_frequency"}
