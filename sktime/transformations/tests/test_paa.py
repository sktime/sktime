"""PAA transformer test code."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.paa import PAA


# Check that exception is raised for bad frames values.
# input types - string, float, negative int, negative float, empty dict.
# correct input is an integer of 1 or more, but less than or equal to series length.
@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_frames", ["str", 1.2, -1.2, -1, {}, 0, 11])
def test_bad_input_args(bad_frames):
    """Test that exception is raised for bad num levels."""
    X = np.arange(10).T

    if not isinstance(bad_frames, int):
        for attribute in ["frames", "frame_size"]:
            with pytest.raises(TypeError):
                PAA(**{attribute: bad_frames}).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            PAA(bad_frames).fit(X).transform(X)


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "frames,frame_size,expected",
    [
        (1, 0, [4.5]),
        (2, 0, [2, 7]),
        (3, 0, [1.2, 4.5, 7.8]),
        (10, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, 2, [0.5, 2.5, 4.5, 6.5, 8.5]),
        (0, 3, [1, 4, 7, 9]),
        (0, 10, [4.5]),
    ],
)
def test_output_of_transformer(frames, frame_size, expected):
    """Test that the transformer has changed the data correctly."""
    X = np.arange(10).T
    paa = PAA(frames, frame_size)
    res = paa.fit_transform(X)
    want = np.array(expected, dtype=np.float64).T
    np.testing.assert_array_equal(res, want)


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "length,frames,dtype,value",
    [
        (5, 2, np.int64, 2_000_000_000_000_000_000),
        (5, 3, np.int64, 2_000_000_000_000_000_000),
        (5, 2, np.int64, -2_000_000_000_000_000_000),
        (5, 2, np.uint64, 4_000_000_000_000_000_000),
        (4, 2, np.int64, 2_000_000_000_000_000_000),
        (5, 2, np.float64, 2_000_000_000_000_000_000),
    ],
)
def test_paa_integer_overflow(length, frames, dtype, value):
    """Test that PAA preserves constant values without integer overflow."""
    X = np.full((length, 1), value, dtype=dtype)

    result = PAA(frames=frames).fit_transform(X)
    expected = np.full(frames, value, dtype=np.float64)

    np.testing.assert_allclose(result.ravel(), expected)


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_paa_large_nonconstant_integers():
    """Test weighted averaging of large nonconstant integer values."""
    X = np.arange(1, 6, dtype=np.int64) * 10**18
    result = PAA(frames=2).fit_transform(X)

    np.testing.assert_allclose(result.ravel(), [1.8e18, 4.2e18])
