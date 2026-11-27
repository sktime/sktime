"""Tests for the SAXlegacy transformer."""

import warnings

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.dictionary_based import SAXlegacy


@pytest.mark.skipif(
    not run_test_for_class(SAXlegacy),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("alphabet_size", [2, 3, 4])
def test_sax_no_overflow_warning(alphabet_size):
    """SAXlegacy should not warn when comparing float32 patterns to breakpoints.

    The last breakpoint used to be ``sys.float_info.max``, which overflows to
    ``inf`` when cast to float32 and raised a ``RuntimeWarning`` for every word.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 1, 40))

    sax = SAXlegacy(word_length=4, alphabet_size=alphabet_size, window_size=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Xt = sax.fit_transform(X)

    assert Xt.shape[0] == 3


@pytest.mark.skipif(
    not run_test_for_class(SAXlegacy),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sax_create_word_assigns_last_symbol_to_large_values():
    """Values above all finite breakpoints should map to the last symbol."""
    sax = SAXlegacy(word_length=3, alphabet_size=4, window_size=0)
    breakpoints = sax._generate_breakpoints()

    # symbols: -1 -> 0, 0.1 -> 2, 100 -> 3 (above the last finite breakpoint)
    pattern = np.array([-1.0, 0.1, 100.0], dtype=np.float32)
    word = sax._create_word(pattern, breakpoints)

    assert word == (0 << 4) | (2 << 2) | 3
