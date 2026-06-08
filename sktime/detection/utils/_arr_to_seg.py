"""Utilities for detection tasks."""

import numpy as np
import pandas as pd


def arr_to_seg(arr):
    """Convert 1D array-like of integers to segmentation data frame.

    Returns a DataFrame with two columns:

    - ilocs: IntervalIndex with start and end of constant blocks of integers.
    - labels: The value in the block.

    Parameters
    ----------
    arr : (array-like) 1D array-like of integers
        the array to convert to a segmentation DataFrame.
        In "dense" format as returned by ``transform`` methods.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns 'ilocs' and 'labels',
        as returned by ``predict_segments`` methods.

    Examples
    --------
    >>> from sktime.detection.utils._arr_to_seg import arr_to_seg
    >>> arr = [1, 1, 2, 2, 2, 3, 4, 4]
    >>> df = arr_to_seg(arr)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array-like of integers.")

    # Identify block boundaries
    diffs = np.diff(arr)
    block_starts = np.where(diffs != 0)[0] + 1
    block_indices = np.split(np.arange(len(arr)), block_starts)

    # Create intervals and segments
    intervals = [
        pd.Interval(block[0], block[-1] + 1, closed="left") for block in block_indices
    ]
    segments = [arr[block[0]] for block in block_indices]

    # Create DataFrame
    df = pd.DataFrame({"ilocs": pd.IntervalIndex(intervals), "labels": segments})

    return df


def sparse_pts_to_seg(arr):
    """Convert 1D array-like of integers to segmentation data frame.

    Returns a DataFrame with two columns:

    - ilocs: IntervalIndex with start and end of consecutive blocks of integers.
    - segments: The value in the block.

    Parameters
    ----------
    arr : (array-like) 1D array-like of integers
        the array to convert to a segmentation DataFrame.
        In format of predict_points, assumed sorted.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with column 'ilocs',
        as returned by ``predict_segments`` methods.

    Examples
    --------
    >>> from sktime.detection.utils._arr_to_seg import sparse_pts_to_seg
    >>> arr = [1, 2, 3, 7, 8, 9, 42, 43]
    >>> df = sparse_pts_to_seg(arr)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array-like of integers.")

    # Identify block boundaries
    diffs = np.diff(arr)
    block_starts_ref = np.where(diffs > 1)[0] + 1

    block_indices = np.split(np.arange(len(arr)), block_starts_ref)

    # Create intervals and segments
    intervals = [
        pd.Interval(arr[b[0]], arr[b[-1]] + 1, closed="left") for b in block_indices
    ]

    # Create DataFrame
    df = pd.DataFrame({"ilocs": pd.IntervalIndex(intervals)})
    return df
