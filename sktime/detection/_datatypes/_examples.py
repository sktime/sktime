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
    labels = pd.Series([0, 1, 1, 0])
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


def _get_example_segments_3():
    """Generate example 3 for segmentation output.

    - overlapping
    - exhaustive
    - labeled
    """
    segs = pd.IntervalIndex.from_tuples([(0, 3), (2, 5), (3, 7), (7, 10)], "left")
    labels = pd.Series([0, 1, 2, 3])
    return pd.DataFrame({"ilocs": segs, "labels": labels})


def _get_example_points_3():
    """Generate example 3 as points only.

    Corresponds to example 3 for point output.
    """
    pts = [2, 3, 5, 7]
    return pd.DataFrame({"ilocs": pts})
