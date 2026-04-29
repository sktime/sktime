# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for panel transformers of time series augmentation."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series import augmenter as aug


def _load_test_data():
    X, y = load_basic_motions()
    X1 = X.iloc[0, 1]
    return X1


def _calc_checksum(X):
    if isinstance(X, pd.DataFrame):
        checksum = round(sum([sum([sum(x) for x in X[c]]) for c in X.columns]), 6)
    else:
        checksum = round(sum(X), 6)
    return checksum


# Test Data
expected_checksum_X = 17.757893


def test_loaded_data():
    """Test of the loaded motion data."""
    X = _load_test_data()
    assert _calc_checksum(X) == expected_checksum_X


# Test WhiteNoiseAugmenter
expected_checksums_white_noise = [7.373241, -3.01141, 12.565567, 5.86128]


@pytest.mark.skipif(
    not run_test_for_class(aug.WhiteNoiseAugmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "parameter",
    [
        (
            {},
            {"scale": 2},
            {"scale": 0.5},
            {"scale": np.std},
        ),
    ],
)
def test_white_noise(parameter):
    """Test of the White Noise Augmenter."""
    X = _load_test_data()
    checksums = []
    for para in parameter:
        augmenter = aug.WhiteNoiseAugmenter(**para)
        Xt = augmenter.fit_transform(X)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
    assert checksums == expected_checksums_white_noise


# Test ReverseAugmenter
expected_checksum_reverse = 17.757893


@pytest.mark.skipif(
    not run_test_for_class(aug.ReverseAugmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_reverse():
    """Test of the White Noise Augmenter."""
    X = _load_test_data()
    augmenter = aug.ReverseAugmenter()
    Xt = augmenter.fit_transform(X)
    checksum = _calc_checksum(Xt)
    assert checksum == expected_checksum_reverse


# Test InvertAugmenter
expected_checksum_invert = -17.757893


@pytest.mark.skipif(
    not run_test_for_class(aug.InvertAugmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_invert():
    """Test of the White Noise Augmenter."""
    X = _load_test_data()
    augmenter = aug.InvertAugmenter()
    Xt = augmenter.fit_transform(X)
    checksum = _calc_checksum(Xt)
    assert checksum == expected_checksum_invert


# Test RandomSamplesAugmenter
expected_checksums_random_samples = [
    17.757893,
    13.588425,
    -0.121451,
    10.684428,
    -1.350097,
]


@pytest.mark.skipif(
    not run_test_for_class(aug.RandomSamplesAugmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "parameter",
    [
        (
            {},
            {"n": 0.5, "without_replacement": True},
            {"n": 2, "without_replacement": True},
            {"n": 1.5, "without_replacement": False},
            {"n": 3, "without_replacement": False},
        ),
    ],
)
def test_random_samples(parameter):
    """Test of the White Noise Augmenter."""
    X = _load_test_data()
    checksums = []
    for para in parameter:
        augmenter = aug.RandomSamplesAugmenter(**para)
        Xt = augmenter.fit_transform(X)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
    assert checksums == expected_checksums_random_samples
