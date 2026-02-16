# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common private utilities for checkers and converters."""

__author__ = ["fkiraly"]


def _metadata_requested(return_metadata):
    """Return whether some metadata has been requested."""
    return not isinstance(return_metadata, bool) or return_metadata


def _ret(valid, msg, metadata, return_metadata):
    """Return switch for checker functions."""
    if _metadata_requested(return_metadata):
        return valid, msg, metadata
    else:
        return valid


def _req(key, return_metadata):
    """Return whether metadata key is requested, boolean."""
    if isinstance(return_metadata, bool):
        return return_metadata
    elif isinstance(return_metadata, str) and not isinstance(key, list):
        return return_metadata == key
    elif isinstance(return_metadata, str) and isinstance(key, list):
        return return_metadata in key
    elif isinstance(return_metadata, list) and not isinstance(key, list):
        return key in return_metadata
    elif isinstance(return_metadata, list) and isinstance(key, list):
        return len(set(key).intersection(return_metadata)) > 0
    else:
        return False


def _wr(d, key, val, return_metadata):
    """Metadata write switch for checker functions."""
    if _req(key, return_metadata):
        d[key] = val

    return d
