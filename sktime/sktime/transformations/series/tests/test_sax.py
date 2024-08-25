"""SAX transformer test code."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.sax import SAX


# Check that exception is raised for bad size values.
# input types - string, float, negative float, empty dict.
# correct input is an integer, see below for value restrictions.
@pytest.mark.skipif(
    not run_test_for_class(SAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_size", ["str", 1.2, -1.2, {}])
def test_bad_input_types(bad_size):
    """Test that exception is raised for bad sizes."""
    for attribute in ["word_size", "alphabet_size", "frame_size"]:
        with pytest.raises(TypeError):
            SAX(**{attribute: bad_size})


@pytest.mark.skipif(
    not run_test_for_class(SAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "attribute,bad_size",
    [
        ("word_size", 0),
        ("alphabet_size", 1),
        ("frame_size", -1),
    ],
)
def test_bad_input_values(attribute, bad_size):
    """Test that word_size is at least 1 (if frame_size is not set)."""
    with pytest.raises(ValueError):
        SAX(**{attribute: bad_size})


@pytest.mark.skipif(
    not run_test_for_class(SAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "word_size,alphabet_size,frame_size,expected",
    [
        (2, 5, 0, [0, 4]),
        (3, 5, 0, [0, 2, 4]),
        (0, 5, 2, [0, 1, 2, 3, 4]),
        (0, 5, 3, [0, 2, 4, 4]),
    ],
)
def test_output_of_transformer(word_size, alphabet_size, frame_size, expected):
    """Test that the transformer has changed the data correctly."""
    X = np.arange(10).T
    sax = SAX(word_size, alphabet_size, frame_size)
    res = sax.fit_transform(X)
    want = np.array(expected, dtype=np.float64).T
    np.testing.assert_array_equal(res, want)
