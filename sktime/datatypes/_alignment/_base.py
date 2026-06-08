# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypeAlignment(BaseDatatype):
    """Alignment data type. Represents an alignment of two or multiple time series.

    Parameters
    ----------
    is_multiple : boolean
        True iff is multiple alignment, i.e., has multiple (3 or more) time series
    """

    _tags = {
        "scitype": "Alignment",
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self, is_multiple=None):
        self.is_multiple = is_multiple

        super().__init__()
