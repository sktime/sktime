"""Isolated numba imports for _sfa."""

__author__ = ["MatthewMiddlehurst", "patrickzib"]

import math

import numpy as np

from sktime.utils.numba.njit import njit

# The binning methods to use: equi-depth, equi-width, information gain or kmeans
binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans"}

# TODO remove imag-part from dc-component component


@njit(fastmath=True, cache=True)
def _discrete_fourier_transform(
    series,
    dft_length,
    norm,
    inverse_sqrt_win_size,
    lower_bounding,
    apply_normalising_factor=True,
    cut_start_if_norm=True,
):
    """Perform a discrete fourier transform using standard O(n^2) transform.

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
    start = 2 if norm else 0
    output_length = start + dft_length

    if cut_start_if_norm:
        c = int(start / 2)
    else:
        c = 0
        start = 0

    dft = np.zeros(output_length - start)
    for i in range(c, int(output_length / 2)):
        for n in range(len(series)):
            dft[(i - c) * 2] += series[n] * math.cos(2 * math.pi * n * i / len(series))
            dft[(i - c) * 2 + 1] += -series[n] * math.sin(
                2 * math.pi * n * i / len(series)
            )

    if apply_normalising_factor:
        if lower_bounding:
            dft[1::2] = dft[1::2] * -1  # lower bounding

        std = np.std(series)
        if std == 0:
            std = 1
        dft *= inverse_sqrt_win_size / std

    return dft


@njit(fastmath=True, cache=True)
def _get_phis(window_size, length):
    phis = np.zeros(length)
    for i in range(int(length / 2)):
        phis[i * 2] += math.cos(2 * math.pi * (-i) / window_size)
        phis[i * 2 + 1] += -math.sin(2 * math.pi * (-i) / window_size)
    return phis


@njit(fastmath=True, cache=True)
def _iterate_mft(
    series, mft_data, phis, window_size, stds, transformed, inverse_sqrt_win_size
):
    for i in range(1, len(transformed)):
        for n in range(0, len(mft_data), 2):
            # only compute needed indices
            real = mft_data[n] + series[i + window_size - 1] - series[i - 1]
            imag = mft_data[n + 1]
            mft_data[n] = real * phis[n] - imag * phis[n + 1]
            mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag

        normalising_factor = inverse_sqrt_win_size / stds[i]
        transformed[i] = mft_data * normalising_factor


@njit(fastmath=True, cache=True)
def _add_level(word, start, level, window_ind, window_size, series_length, level_bits):
    num_quadrants = pow(2, level)
    quadrant = start + int(
        (window_ind + int(window_size / 2)) / int(series_length / num_quadrants)
    )
    return (word << level_bits) | quadrant, num_quadrants


@njit(fastmath=True, cache=True)
def _add_level_typed(word, start, level, window_ind, window_size, series_length):
    num_quadrants = pow(2, level)
    quadrant = start + int(
        (window_ind + int(window_size / 2)) / int(series_length / num_quadrants)
    )
    return (word, quadrant), num_quadrants


@njit(fastmath=True, cache=True)
def _create_word(dft, word_length, alphabet_size, breakpoints, letter_bits):
    word = np.int64(0)
    for i in range(word_length):
        for bp in range(alphabet_size):
            if dft[i] <= breakpoints[i][bp]:
                word = (word << letter_bits) | bp
                break

    return word


@njit(fastmath=True, cache=True)
def _calc_incremental_mean_std(series, end, window_size):
    stds = np.zeros(end)
    window = series[0:window_size]
    series_sum = np.sum(window)
    square_sum = np.sum(np.multiply(window, window))

    r_window_length = 1 / window_size
    mean = series_sum * r_window_length
    buf = math.sqrt(square_sum * r_window_length - mean * mean)
    stds[0] = buf if buf > 1e-8 else 1

    for w in range(1, end):
        series_sum += series[w + window_size - 1] - series[w - 1]
        mean = series_sum * r_window_length
        square_sum += (
            series[w + window_size - 1] * series[w + window_size - 1]
            - series[w - 1] * series[w - 1]
        )
        buf = math.sqrt(square_sum * r_window_length - mean * mean)
        stds[w] = buf if buf > 1e-8 else 1

    return stds


@njit(fastmath=True, cache=True)
def _create_bigram_word(word, other_word, word_bits):
    return (word << word_bits) | other_word


@njit(fastmath=True, cache=True)
def _shorten_word(word, amount, letter_bits):
    # shorten a word by set amount of letters
    return word >> amount * letter_bits
