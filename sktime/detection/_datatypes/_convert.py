"""Utilities to handle checks and conversions between output formats of detectors."""

import numpy as np
import pandas as pd


def _convert_points_to_segments(points_df, len_X=None, include_labels=False):
    """Convert points-like output to segments-like output.

    Parameters
    ----------
    points_df : pd.DataFrame
        Points-like output of a detector
    len_X : int, optional, default=None
        Length of the input data.
        If None, the last segment is assumed to be length 1.
    include_labels : bool, optional, default=False
        Whether to include labels in the output.
        If True, labels are RangeIndex of the same length as the segments.

    Returns
    -------
    pd.DataFrame
        Segments-like output of a detector
    """
    if len(points_df) == 0:
        return pd.DataFrame(columns=["ilocs", "labels"], dtype="int64")

    points = points_df.ilocs.values
    if points[0] != 0:
        points = np.insert(points, 0, 0)
    if len_X is None:
        points = np.append(points, points[-1] + 1)
    elif points[-1] != len_X:
        points = np.append(points, len_X)

    ilocs = pd.IntervalIndex.from_breaks(points, closed="left")

    df_dict = {"ilocs": ilocs}
    if include_labels:
        labels = pd.RangeIndex(len(ilocs))
        df_dict["labels"] = labels

    return pd.DataFrame(df_dict)


def _convert_segments_to_points(seg_df, len_X=None):
    """Convert points-like output to segments-like output.

    Parameters
    ----------
    seg_df : pd.DataFrame
        Points-like output of a detector
    len_X : int, optional
        Length of the input data.
        If None, the (right point) end of the last segment is omitted.

    Returns
    -------
    pd.DataFrame
        Segments-like output of a detector
    """
    if len(seg_df) == 0:
        return pd.DataFrame(columns=["ilocs"], dtype="int64")

    ix = seg_df.set_index("ilocs").index
    vals = np.array([ix.left.values, ix.right.values]).transpose().flatten()
    vals = np.unique(vals)

    if not ix.is_non_overlapping_monotonic:
        vals = np.sort(vals)

    if vals[0] == 0:
        vals = vals[1:]

    if len_X is None or vals[-1] == len_X:
        vals = vals[:-1]

    points_df = pd.DataFrame({"ilocs": vals})
    return points_df
