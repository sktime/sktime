"""Utilities to handle checks and conversions between output formats of detectors."""

import pandas as pd


def _is_valid_detection(obj, type="points"):
    """Check if the input is valid common output format.

    Parameters
    ----------
    obj : pd.DataFrame
        Output of a detector to check for validity
    type : str {"points", "segments"}
        Type of output to check for validity
    """
    if not isinstance(obj, pd.DataFrame):
        return False
    if not isinstance(obj.index, pd.RangeIndex):
        return False
    if "ilocs" not in obj.columns:
        return False

    if type == "points":
        return _is_points_dtype(obj)
    elif type == "segments":
        return _is_segments_dtype(obj)

    return False


def _is_points_dtype(obj):
    """Check if the input is points-like.

    Assumes validity of input is checked elsewhere.

    Parameters
    ----------
    obj : pd.DataFrame
        Output of a detector to check
    """
    return pd.api.types.is_integer_dtype(obj.ilocs.dtype)


def _is_segments_dtype(obj):
    """Check if the input is valid segments-like.

    Assumes validity of input is checked elsewhere.

    Parameters
    ----------
    obj : pd.DataFrame
        Output of a detector to check
    """
    ilocs_dtype = obj.ilocs.dtype
    if not (isinstance(ilocs_dtype, pd.IntervalDtype) or len(obj) == 0):
        return False
    if not len(obj) == 0:
        return pd.api.types.is_integer_dtype(ilocs_dtype.subtype)
    return True
