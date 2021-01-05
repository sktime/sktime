# -*- coding: utf-8 -*-
import sys

import numpy as np

from sktime.datasets import load_gunpoint
from sktime.transformations.panel.dictionary_based._sfa import SFA
from sktime.utils.data_processing import from_nested_to_2d_array


# Check the transformer has changed the data correctly.
def test_transformer():
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        anova=False,
        binning_method="equi-depth",
    ).fit(X, y)

    # print("Equi Depth")
    # print(p.breakpoints)

    assert p.breakpoints.shape == (word_length, alphabet_size)
    assert np.equal(0, p.breakpoints[1, :-1]).all()  # imag component is 0
    _ = p.transform(X, y)

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        anova=False,
        binning_method="equi-width",
    ).fit(X, y)

    # print("Equi Width")
    # print(p.breakpoints)

    assert p.breakpoints.shape == (word_length, alphabet_size)
    assert np.equal(0, p.breakpoints[1, :-1]).all()  # imag component is 0
    _ = p.transform(X, y)

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        anova=False,
        binning_method="information-gain",
    ).fit(X, y)
    # print("Information Gain")
    # print(p.breakpoints)

    assert p.breakpoints.shape == (word_length, alphabet_size)

    # print(p.breakpoints[1, :-1])
    assert np.equal(sys.float_info.max, p.breakpoints[1, :-1]).all()
    _ = p.transform(X, y)


def test_dft_mft():
    # load training data
    X, Y = load_gunpoint(split="train", return_X_y=True)
    X_tab = from_nested_to_2d_array(X, return_numpy=True)

    word_length = 6
    alphabet_size = 4

    # print("Single DFT transformation")
    window_size = np.shape(X_tab)[1]
    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        window_size=window_size,
        binning_method="equi-depth",
    ).fit(X, Y)
    dft = p._discrete_fourier_transform(X_tab[0])
    mft = p._mft(X_tab[0])

    assert (mft - dft < 0.0001).all()

    # print("Windowed DFT transformation")

    for norm in [True, False]:
        for window_size in [140]:
            p = SFA(
                word_length=word_length,
                norm=norm,
                alphabet_size=alphabet_size,
                window_size=window_size,
                binning_method="equi-depth",
            ).fit(X, Y)
            mft = p._mft(X_tab[0])
            for i in range(len(X_tab[0]) - window_size + 1):
                dft_transformed = p._discrete_fourier_transform(
                    X_tab[0, i : window_size + i]
                )
                assert (mft[i] - dft_transformed < 0.001).all()

            assert len(mft) == len(X_tab[0]) - window_size + 1
            assert len(mft[0]) == word_length


def test_sfa_anova():
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    for binning in ["information-gain", "equi-depth"]:
        # print("SFA with ANOVA one-sided test")
        window_size = 32
        p = SFA(
            word_length=word_length,
            anova=True,
            alphabet_size=alphabet_size,
            window_size=window_size,
            binning_method=binning,
        ).fit(X, y)

        # print(p.breakpoints)
        # print(p.support)
        # print(p.dft_length)

        assert p.breakpoints.shape == (word_length, alphabet_size)
        _ = p.transform(X, y)

        # print("SFA with first feq coefficients")
        p2 = SFA(
            word_length=word_length,
            anova=False,
            alphabet_size=alphabet_size,
            window_size=window_size,
            binning_method=binning,
        ).fit(X, y)

        # print(p2.breakpoints)
        # print(p2.support)
        # print(p2.dft_length)

        assert p.dft_length != p2.dft_length
        assert (p.breakpoints != p2.breakpoints).any()
        _ = p2.transform(X, y)


# test word lengths larger than the window-length
def test_word_lengths():
    # load training data
    X, y = load_gunpoint(split="train", return_X_y=True)

    word_lengths = [6, 7]
    alphabet_size = 4
    window_sizes = [5, 6]

    try:
        for binning in ["equi-depth", "information-gain"]:
            for word_length in word_lengths:
                for bigrams in [True, False]:
                    for norm in [True, False]:
                        for anova in [True, False]:
                            for window_size in window_sizes:
                                p = SFA(
                                    word_length=word_length,
                                    anova=anova,
                                    alphabet_size=alphabet_size,
                                    bigrams=bigrams,
                                    window_size=window_size,
                                    norm=norm,
                                    binning_method=binning,
                                ).fit(X, y)

                                # print("Norm", norm, "Anova", anova)
                                # print(np.shape(p.breakpoints), word_length,
                                #      window_size)
                                # print("dft_length", p.dft_length,
                                #      "word_length", p.word_length)
                                assert p.breakpoints is not None

                                _ = p.transform(X, y)

    except Exception as err:
        raise AssertionError("An unexpected exception {0} raised.".format(repr(err)))


# def test_reproducability():
#     # load training data
#     X, y = load_gunpoint(split="train", return_X_y=True)
#     m = len(X.iloc[0]['dim_0'])
#
#     p = SFA(word_length=4,
#             anova=True,
#             alphabet_size=4,
#             bigrams=False,
#             window_size=m,
#             norm=True,
#             lower_bounding=False,
#             binning_method="equi-depth").fit(X, y)
#
#     print(p.breakpoints)
#     print(p.support)
#
#     print("m", m)
#     p = SFA(word_length=4,
#             anova=True,
#             alphabet_size=4,
#             bigrams=False,
#             window_size=10,
#             norm=False,
#             lower_bounding=False,
#             binning_method="equi-width").fit(X, y)
#
#     print(p.breakpoints)
#     print(p.support)
