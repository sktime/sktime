# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for WaveletPacketTransformer."""

__author__ = ["ved197338"]

import numpy as np
import pandas as pd
import pytest

from sktime.transformations.wavelet_packet import WaveletPacketTransformer


def test_energy_output_shape():
    """Level-2 decomposition should produce 4 energy features."""
    series = pd.Series(np.sin(np.linspace(0, 10, 64)), name="val")
    result = WaveletPacketTransformer(level=2, output_feature="energy").fit_transform(
        series
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 4)
    assert all("energy" in c for c in result.columns)


def test_entropy_output_shape():
    """Level-1 decomposition should produce 2 entropy features."""
    rng = np.random.RandomState(42)
    series = pd.Series(rng.randn(32), name="noise")
    result = WaveletPacketTransformer(level=1, output_feature="entropy").fit_transform(
        series
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    assert all("entropy" in c for c in result.columns)


def test_coefficients_preserve_length():
    """Coefficient output should have the same length as the input."""
    series = pd.Series(np.arange(16, dtype=float), name="ramp")
    result = WaveletPacketTransformer(
        level=2, output_feature="coefficients"
    ).fit_transform(series)

    assert isinstance(result, pd.Series)
    assert len(result) == 16


def test_rejects_negative_level():
    """Negative level should raise ValueError."""
    with pytest.raises(ValueError, match="non-negative integer"):
        t = WaveletPacketTransformer(level=-1)
        t.fit_transform(pd.Series(np.arange(8)))


def test_rejects_unknown_output_feature():
    """Unknown output_feature should raise ValueError."""
    with pytest.raises(ValueError, match="output_feature"):
        t = WaveletPacketTransformer(output_feature="invalid")
        t.fit_transform(pd.Series(np.arange(8)))


def test_odd_length_signal_handling():
    """Odd length inputs should be handled gracefully without dropping samples."""
    series = pd.Series(np.arange(15, dtype=float), name="odd")
    result = WaveletPacketTransformer(
        level=1, output_feature="coefficients"
    ).fit_transform(series)
    assert len(result) == 15


def test_excessive_decomposition_level_raises():
    """Level greater than log2(n_samples) should raise ValueError."""
    series = pd.Series(np.arange(8, dtype=float))
    with pytest.raises(ValueError, match="exceeds maximum supported"):
        WaveletPacketTransformer(level=10).fit_transform(series)


def test_haar_wavelet_energy_conservation():
    """Sum of node energies should equal total energy for Haar wavelet."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    series = pd.Series(x)
    result = WaveletPacketTransformer(level=1, output_feature="energy").fit_transform(
        series
    )
    total_node_energy = result.iloc[0].sum()
    original_energy = np.sum(x**2)
    np.testing.assert_allclose(total_node_energy, original_energy, rtol=1e-5)
