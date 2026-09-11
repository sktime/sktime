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
