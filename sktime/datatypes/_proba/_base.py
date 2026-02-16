# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypeProba(BaseDatatype):
    """Probabilistic forecast data type.

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    """

    _tags = {
        "scitype": "Alignment",
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self, is_univariate=None, is_empty=None, has_nans=None):
        self.is_univariate = is_univariate
        self.is_empty = is_empty
        self.has_nans = has_nans

        super().__init__()
