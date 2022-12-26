# -*- coding: utf-8 -*-
"""Tests for SFA utilities."""

import sys

import numpy as np
import pytest

from sktime.datasets import load_gunpoint
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.transformations.panel.dictionary_based._sfa import SFA
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
# Check the transformer has changed the data correctly.
@pytest.mark.parametrize(
    "binning_method", ["equi-depth", "equi-width", "information-gain", "kmeans"]
)
def test_transformer(binning_method):
    """Test SFA transformer expected output."""
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        binning_method=binning_method,
    )
    p.fit(X, y)

    assert p.breakpoints.shape == (word_length, alphabet_size)
    i = sys.float_info.max if binning_method == "information-gain" else 0
    assert np.equal(i, p.breakpoints[1, :-1]).all()  # imag component is 0 or inf
    _ = p.transform(X, y)


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("use_fallback_dft", [True, False])
@pytest.mark.parametrize("norm", [True, False])
def test_dft_mft(use_fallback_dft, norm):
    """Test DFT and MFT functions."""
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)
    X_tab = from_nested_to_2d_array(X, return_numpy=True)

    word_length = 6
    alphabet_size = 4

    # Single DFT transformation
    window_size = np.shape(X_tab)[1]

    p = SFA(
        word_length=6,
        alphabet_size=4,
        window_size=window_size,
        norm=norm,
        use_fallback_dft=use_fallback_dft,
    ).fit(X, y)

    if use_fallback_dft:
        from sktime.transformations.panel.dictionary_based._sfa_numba import (
            _discrete_fourier_transform,
        )

        dft = _discrete_fourier_transform(X_tab[0], word_length, norm, 1, True)
    else:
        dft = p._fast_fourier_transform(X_tab[0])

    mft = p._mft(X_tab[0])

    assert (mft - dft < 0.0001).all()

    # Windowed DFT transformation
    window_size = 140

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        window_size=window_size,
        norm=norm,
        use_fallback_dft=use_fallback_dft,
    ).fit(X, y)

    mft = p._mft(X_tab[0])
    for i in range(len(X_tab[0]) - window_size + 1):
        if use_fallback_dft:
            dft = _discrete_fourier_transform(
                X_tab[0, i : window_size + i], word_length, norm, 1, True
            )
        else:
            dft = p._fast_fourier_transform(X_tab[0, i : window_size + i])

        assert (mft[i] - dft < 0.001).all()

    assert len(mft) == len(X_tab[0]) - window_size + 1
    assert len(mft[0]) == word_length


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("binning_method", ["equi-depth", "information-gain"])
def test_sfa_anova(binning_method):
    """Test SFA expected breakpoints."""
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    # SFA with ANOVA one-sided test
    window_size = 32
    p = SFA(
        word_length=word_length,
        anova=True,
        alphabet_size=alphabet_size,
        window_size=window_size,
        binning_method=binning_method,
    ).fit(X, y)

    assert p.breakpoints.shape == (word_length, alphabet_size)
    _ = p.transform(X, y)

    # SFA with first feq coefficients
    p2 = SFA(
        word_length=word_length,
        anova=False,
        alphabet_size=alphabet_size,
        window_size=window_size,
        binning_method=binning_method,
    ).fit(X, y)

    assert p.dft_length != p2.dft_length
    assert (p.breakpoints != p2.breakpoints).any()
    _ = p2.transform(X, y)


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
# test word lengths larger than the window-length
@pytest.mark.parametrize("word_length", [6, 7])
@pytest.mark.parametrize("alphabet_size", [4, 5])
@pytest.mark.parametrize("window_size", [5, 6])
@pytest.mark.parametrize("bigrams", [True, False])
@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("use_fallback_dft", [True, False])
def test_word_lengths(
    word_length, alphabet_size, window_size, bigrams, levels, use_fallback_dft
):
    """Test expected word lengths."""
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        window_size=window_size,
        bigrams=bigrams,
        levels=levels,
        use_fallback_dft=use_fallback_dft,
    ).fit(X, y)

    assert p.breakpoints is not None
    _ = p.transform(X, y)


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_bit_size():
    """Test expected bit size on training data."""
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_length = 40
    alphabet_size = 12
    window_size = 75

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        levels=2,
        bigrams=True,
        window_size=window_size,
    ).fit(X, y)

    w = p.transform(X)
    lengths = [x.bit_length() for x in list(w[0][0].keys())]

    assert np.min(lengths) > 128
    assert len(p.word_list(list(w[0][0].keys())[0])[0]) == 40


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_typed_dict():
    """Test word list from typed dict."""
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        levels=2,
        typed_dict=True,
    )
    p.fit(X, y)
    word_list = p.bag_to_string(p.transform(X, y)[0][0])

    word_length = 6
    alphabet_size = 4

    p2 = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        levels=2,
        typed_dict=False,
    )
    p2.fit(X, y)
    word_list2 = p2.bag_to_string(p2.transform(X, y)[0][0])

    assert word_list == word_list2
