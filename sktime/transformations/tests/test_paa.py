"""PAA transformer test code."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.paa import PAA


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
