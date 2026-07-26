"""Utilities for detection tasks."""

import numpy as np
import pandas as pd


def seg_middlepoint(seg_df):
    """Convert segmentation data frame to midpoint detections.

    Parameters
    ----------
    seg_df : pd.DataFrame
        data frame in segmentation format, with columns:

        * ``ilocs`` - IntervalIndex with start and end of consecutive blocks of integers
        * optional: ``labels`` - The label value of the block

    Returns
    -------
    point_df: pd.DataFrame
        data frame in point detection format, with columns:

        * ``ilocs`` - integer index of the midpoint of each block
        * optional: ``labels`` - The label value of the block

        Return has ``labels`` column iff input has ``labels`` column.

    Examples
    --------
    >>> from sktime.detection._datatypes._examples import _get_example_segments_1
    >>> from sktime.detection.utils._seg_middle import seg_middlepoint
    >>> seg_df = _get_example_segments_1()
    >>> point_df = seg_middlepoint(seg_df)
    """
    ilocs_mid = seg_df.ilocs.apply(lambda x: x.mid)
    # round down to integer
    ilocs_mid = ilocs_mid.apply(np.floor).astype(int)

    point_data = {"ilocs": ilocs_mid}

    if "labels" in seg_df.columns:
        point_data["labels"] = seg_df.labels

    point_df = pd.DataFrame(point_data)
    return point_df
