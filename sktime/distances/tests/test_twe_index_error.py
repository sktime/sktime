import numpy as np

from sktime.distances import twe_alignment_path


def test_twe_alignment_path_default_bounding_matrix():
    """Test twe_alignment_path with default bounding matrix to avoid IndexError.
    This test ensures that the twe_alignment_path function works correctly
    when no bounding matrix is provided, preventing any IndexError."""
    x = np.array([0, 1, 1, 2, 1, 1, 0])
    y = np.array([1, 1, 2, 1, 1, 0, 0])

    path, dist = twe_alignment_path(x, y)

    assert path[0] == (0, 0)
    assert path[-1] == (len(x) - 1, len(y) - 1)
