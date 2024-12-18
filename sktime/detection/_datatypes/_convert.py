"""Utilities to handle checks and conversions between output formats of detectors."""

import numpy as np
import pandas as pd


def _convert_points_to_segments(points_df, len_X=None, include_labels=False):
    """Convert points-like output to segments-like output.

    Parameters
    ----------
    points_df : pd.DataFrame
        Points-like output of a detector
    len_X : int, optional
        Length of the input data.
        If None, the last segment is assumed to be length 1.

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
