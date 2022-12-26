# -*- coding: utf-8 -*-
"""Isolated numba imports for _sfa_fast."""


__author__ = ["patrickzib"]

import math
from warnings import simplefilter

import numpy as np

from sktime.utils.numba.njit import njit
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("numba", severity="none"):
    from numba import (
        NumbaPendingDeprecationWarning,
        NumbaTypeSafetyWarning,
        objmode,
        prange,
    )
    from numba.core import types
    from numba.typed import Dict

    simplefilter(action="ignore", category=NumbaPendingDeprecationWarning)
    simplefilter(action="ignore", category=NumbaTypeSafetyWarning)

# The binning methods to use: equi-depth, equi-width, information gain or kmeans
binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans", "quantile"}


@njit(fastmath=True, cache=True)
def _binning_dft(
    X,
    window_size,
    series_length,
    dft_length,
    norm,
    inverse_sqrt_win_size,
    lower_bounding,
):
    num_windows_per_inst = math.ceil(series_length / window_size)

    # Splits individual time series into windows and returns the DFT for each
    data = np.zeros((len(X), num_windows_per_inst, window_size))
    for i in prange(len(X)):
        for j in range(num_windows_per_inst - 1):
            data[i, j] = X[i, window_size * j : window_size * (j + 1)]

        start = series_length - window_size
        data[i, -1] = X[i, start:series_length]

    dft = np.zeros((len(X), num_windows_per_inst, dft_length))
    for i in prange(len(X)):
        return_val = _fast_fourier_transform(
            data[i], norm, dft_length, inverse_sqrt_win_size
        )
        dft[i] = return_val

    if lower_bounding:
        dft[:, :, 1::2] = dft[:, :, 1::2] * -1  # lower bounding

    return dft.reshape(dft.shape[0] * dft.shape[1], dft_length)


@njit(fastmath=True, cache=True)
def _fast_fourier_transform(X, norm, dft_length, inverse_sqrt_win_size):
    """Perform a discrete fourier transform using the fast fourier transform.

    if self.norm is True, then the first term of the DFT is ignored

    Input
    -------
    X : The training input samples.  array-like or sparse matrix of
    shape = [n_samps, num_atts]

    Returns
    -------
    1D array of fourier term, real_0,imag_0, real_1, imag_1 etc, length
    num_atts or
    num_atts-2 if if self.norm is True
    """
    # first two are real and imaginary parts
    start = 2 if norm else 0
    length = start + dft_length
    dft = np.zeros((len(X), length))  # , dtype=np.float64

    stds = np.zeros(len(X))
    for i in range(len(stds)):
        stds[i] = np.std(X[i])
    # stds = np.std(X, axis=1)  # not available in numba
    stds = np.where(stds < 1e-8, 1e-8, stds)

    with objmode(X_ffts="complex128[:,:]"):
        X_ffts = np.fft.rfft(X, axis=1)  # complex128
    reals = np.real(X_ffts)  # float64[]
    imags = np.imag(X_ffts)  # float64[]
    dft[:, 0::2] = reals[:, 0 : length // 2]
    dft[:, 1::2] = imags[:, 0 : length // 2]
    dft /= stds.reshape(-1, 1)
    dft *= inverse_sqrt_win_size

    return dft[:, start:]


@njit(fastmath=True, cache=True)
def _transform_case(
    X,
    window_size,
    dft_length,
    word_length,
    norm,
    remove_repeat_words,
    support,
    anova,
    variance,
    breakpoints,
    letter_bits,
    bigrams,
    skip_grams,
    inverse_sqrt_win_size,
    lower_bounding,
):
    dfts = _mft(
        X,
        window_size,
        dft_length,
        norm,
        support,
        anova,
        variance,
        inverse_sqrt_win_size,
        lower_bounding,
    )

    words = generate_words(
        dfts,
        bigrams,
        skip_grams,
        window_size,
        breakpoints,
        word_length,
        letter_bits,
    )

    if remove_repeat_words:
        words = remove_repeating_words(words)

    return words, dfts


@njit(fastmath=True, cache=True)
def remove_repeating_words(words):
    """Remove repeating words."""
    for i in range(words.shape[0]):
        last_word = 0
        for j in range(words.shape[1]):
            if last_word == words[i, j]:
                # We encode the repeated words as 0 and remove them
                # This is implementged using np.nonzero in numba. Thus must be 0
                words[i, j] = 0
            last_word = words[i, j]

    return words


@njit(fastmath=True, cache=True)
def _calc_incremental_mean_std(series, end, window_size):
    stds = np.zeros(end)
    window = series[0:window_size]
    series_sum = np.sum(window)
    square_sum = np.sum(np.multiply(window, window))

    r_window_length = 1.0 / window_size
    mean = series_sum * r_window_length
    buf = math.sqrt(max(square_sum * r_window_length - mean * mean, 0.0))
    stds[0] = buf if buf > 1e-8 else 1e-8

    for w in range(1, end):
        series_sum += series[w + window_size - 1] - series[w - 1]
        mean = series_sum * r_window_length
        square_sum += (
            series[w + window_size - 1] * series[w + window_size - 1]
            - series[w - 1] * series[w - 1]
        )
        buf = math.sqrt(max(square_sum * r_window_length - mean * mean, 0.0))
        stds[w] = buf if buf > 1e-8 else 1e-8

    return stds


@njit(fastmath=True, cache=True)
def _get_phis(window_size, length):
    phis = np.zeros(length)
    i = np.arange(length // 2)
    const = 2 * np.pi / window_size
    phis[0::2] = np.cos((-i) * const)
    phis[1::2] = -np.sin((-i) * const)
    return phis


@njit(fastmath=True, cache=True)
def generate_words(
    dfts, bigrams, skip_grams, window_size, breakpoints, word_length, letter_bits
):
    """Generate words."""
    needed_size = dfts.shape[1]
    if bigrams:
        # allocate memory for bigrams
        needed_size += max(0, dfts.shape[1] - window_size)
    if skip_grams:
        # allocate memory for 2- and 3-skip-grams
        needed_size += max(0, 2 * dfts.shape[1] - 5 * window_size)

    words = np.zeros((dfts.shape[0], needed_size), dtype=np.uint32)

    letter_bits = np.uint32(letter_bits)
    word_bits = word_length * letter_bits  # dfts.shape[2] * letter_bits

    # special case: binary breakpoints
    if breakpoints.shape[1] == 2:
        vector = np.zeros((breakpoints.shape[0]), dtype=np.float32)
        for i in range(breakpoints.shape[0]):
            vector[i] = breakpoints.shape[1] ** i

        for a in prange(dfts.shape[0]):
            match = (dfts[a] <= breakpoints[:, 0]).astype(np.float32)
            words[a, : dfts.shape[1]] = np.dot(match, vector).astype(np.uint32)

    # general case: alphabet-size many breakpoints
    else:
        for a in prange(dfts.shape[0]):
            for i in range(word_length):  # range(dfts.shape[2]):
                words[a, : dfts.shape[1]] = (
                    words[a, : dfts.shape[1]] << letter_bits
                ) | np.digitize(dfts[a, :, i], breakpoints[i], right=True)

    # add bigrams
    if bigrams:
        for a in prange(0, dfts.shape[1] - window_size):
            first_word = words[:, a]
            second_word = words[:, a + window_size]
            words[:, dfts.shape[1] + a] = (first_word << word_bits) | second_word

    # # add 2,3-skip-grams
    # if skip_grams:
    #     for s in range(2, 4):
    #         for a in range(0, dfts.shape[1] - s * window_size):
    #             first_word = words[:, a]
    #             second_word = words[:, a + s * window_size]
    #             words[:, dfts.shape[1] + a] = (first_word << word_bits) | second_word

    return words


@njit(cache=True, fastmath=True)
def create_feature_names(sfa_words):
    """Create feature names."""
    feature_names = set()
    for t_words in sfa_words:
        for t_word in t_words:
            feature_names.add(t_word)
    return feature_names


@njit(fastmath=True, cache=True)
def _mft(
    X,
    window_size,
    dft_length,
    norm,
    support,
    anova,
    variance,
    inverse_sqrt_win_size,
    lower_bounding,
):
    start_offset = 2 if norm else 0
    length = dft_length + start_offset + dft_length % 2
    end = max(1, len(X[0]) - window_size + 1)

    #  compute mask for only those indices needed and not all indices
    if anova or variance:
        support = support + start_offset
        indices = np.full(length, False)
        mask = np.full(length, False)
        for s in support:
            indices[s] = True
            mask[s] = True
            if (s % 2) == 0:  # even
                indices[s + 1] = True
            else:  # uneven
                indices[s - 1] = True
        mask = mask[indices]
    else:
        indices = np.full(length, True)

    phis = _get_phis(window_size, length)
    transformed = np.zeros((X.shape[0], end, length))

    # 1. First run using DFT
    with objmode(X_ffts="complex128[:,:]"):
        X_ffts = np.fft.rfft(X[:, :window_size], axis=1)  # complex128
    reals = np.real(X_ffts)  # float64[]
    imags = np.imag(X_ffts)  # float64[]
    transformed[:, 0, 0::2] = reals[:, 0 : length // 2]
    transformed[:, 0, 1::2] = imags[:, 0 : length // 2]

    # 2. Other runs using MFT
    # X2 = X.reshape(X.shape[0], X.shape[1], 1)
    # Bugfix to allow for slices on original X like in TEASER
    X2 = X.copy().reshape(X.shape[0], X.shape[1], 1)

    # compute only those indices needed and not all
    phis2 = phis[indices]
    transformed2 = transformed[:, :, indices]
    for i in range(1, end):
        reals = transformed2[:, i - 1, 0::2] + X2[:, i + window_size - 1] - X2[:, i - 1]
        imags = transformed2[:, i - 1, 1::2]
        transformed2[:, i, 0::2] = (
            reals * phis2[:length:2] - imags * phis2[1 : (length + 1) : 2]
        )
        transformed2[:, i, 1::2] = (
            reals * phis2[1 : (length + 1) : 2] + phis2[:length:2] * imags
        )

    transformed2 = transformed2 * inverse_sqrt_win_size

    if lower_bounding:
        transformed2[:, :, 1::2] = transformed2[:, :, 1::2] * -1

    # compute STDs
    stds = np.zeros((X.shape[0], end))
    for a in range(X.shape[0]):
        stds[a] = _calc_incremental_mean_std(X[a], end, window_size)

    # divide all by stds and use only the best indices
    if anova or variance:
        return transformed2[:, :, mask] / stds.reshape(stds.shape[0], stds.shape[1], 1)
    else:
        return (transformed2 / stds.reshape(stds.shape[0], stds.shape[1], 1))[
            :, :, start_offset:
        ]


@njit(cache=True, fastmath=True)
def create_bag_none(
    breakpoints, n_instances, sfa_words, word_length, remove_repeat_words
):
    """Create bag."""
    feature_count = np.uint32(breakpoints.shape[1] ** word_length)
    all_win_words = np.zeros((n_instances, feature_count), dtype=np.uint32)

    for j in prange(sfa_words.shape[0]):
        # this mask is used to encode the repeated words
        if remove_repeat_words:
            masked = np.nonzero(sfa_words[j])
            all_win_words[j, :] = np.bincount(
                sfa_words[j][masked], minlength=feature_count
            )
        else:
            all_win_words[j, :] = np.bincount(sfa_words[j], minlength=feature_count)

    return all_win_words


@njit(cache=True, fastmath=True)
def create_bag_feature_selection(
    n_instances, relevant_features_idx, feature_names, sfa_words, remove_repeat_words
):
    """Create bag, feature selection."""
    relevant_features = Dict.empty(key_type=types.uint32, value_type=types.uint32)
    for k, v in zip(
        feature_names[relevant_features_idx],
        np.arange(len(relevant_features_idx), dtype=np.uint32),
    ):
        relevant_features[k] = v

    if remove_repeat_words:
        if 0 in relevant_features:
            del relevant_features[0]

    all_win_words = np.zeros((n_instances, len(relevant_features_idx)), dtype=np.uint32)
    for j in range(sfa_words.shape[0]):
        for key in sfa_words[j]:
            if key in relevant_features:
                all_win_words[j, relevant_features[key]] += 1
    return all_win_words, relevant_features


@njit(cache=True, fastmath=True)
def create_bag_transform(
    feature_count,
    feature_selection,
    relevant_features,
    sfa_words,
    bigrams,
    remove_repeat_words,
):
    """Create bag, transform."""
    # merging arrays
    all_win_words = np.zeros((len(sfa_words), feature_count), np.uint32)
    for j in prange(sfa_words.shape[0]):
        if len(relevant_features) == 0 and feature_selection == "none":
            # this mask is used to encode the repeated words
            if remove_repeat_words:
                masked = np.nonzero(sfa_words[j])
                all_win_words[j, :] = np.bincount(
                    sfa_words[j][masked], minlength=feature_count
                )
            else:
                all_win_words[j, :] = np.bincount(sfa_words[j], minlength=feature_count)
        else:
            if remove_repeat_words:
                if 0 in relevant_features:
                    del relevant_features[0]

            for _, key in enumerate(sfa_words[j]):
                if key in relevant_features:
                    o = relevant_features[key]
                    all_win_words[j, o] += 1

    return all_win_words, all_win_words.shape[1]


@njit(fastmath=True, cache=True)
def shorten_words(words, amount, letter_bits):
    """Shorten words."""
    new_words = np.zeros((words.shape[0], words.shape[1]), dtype=np.uint32)

    # Unigrams
    shift_len = amount * letter_bits
    for j in prange(words.shape[1]):
        # shorten a word by set amount of letters
        new_words[:, j] = words[:, j] >> shift_len

    # TODO Bigrams
    # if bigrams:
    #     for a in range(0, n_instances):
    #         first_word = new_words[:, a]
    #         second_word = new_words[:, a + window_size]
    #         words[:, n_instances + a] = (first_word << word_bits) | second_word

    return new_words
