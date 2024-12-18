"""Utilities to handle checks and conversions between output formats of detectors."""

import pandas as pd


def _get_example_segments_0():
    """Generate example 0 for segmentation output.

    - non-overlapping
    - non-exhaustive
    - no labels
    """
    segs = pd.IntervalIndex.from_tuples([(1, 2), (3, 6), (7, 10)], "left")
    return pd.DataFrame({"ilocs": segs})


def _get_example_segments_1():
    """Generate example 1 for segmentation output.

    - non-overlapping
    - non-exhaustive
    - labeled
    """
    segs = pd.IntervalIndex.from_tuples([(1, 2), (3, 6), (7, 10), (12, 14)], "left")
    labels = pd.Series([0, 1, 1, 0], index=segs)
    return pd.DataFrame({"ilocs": segs, "labels": labels})


def _get_example_segments_2():
    """Generate example 2 for segmentation output.

    - non-overlapping
    - exhaustive
    - labeled
    """
    segs = pd.IntervalIndex.from_tuples([(0, 2), (2, 3), (3, 7), (7, 10)], "left")
    labels = pd.Series([0, 1, 2, 3])
    return pd.DataFrame({"ilocs": segs, "labels": labels})


def _get_example_points_2():
    """Generate example 2 for point output.

    Corresponds to example 2 for segmentation output.
    """
    points = pd.Series([2, 3, 7], name="ilocs")
    return pd.DataFrame({"ilocs": points})
