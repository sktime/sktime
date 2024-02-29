# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class BaseSeries(BaseDatatype):
    """Base class for Series data types.

    Parameters are inferred by check.
    
    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series  
    """
    _tags = {
        "scitype": "Series",
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(
        self,
        is_univariate=None,
        is_equally_spaced=None,
        is_empty=None,
        has_nans=None,
        n_features=None,
        feature_names=None,
    ):
        self.is_univariate = is_univariate
        self.is_equally_spaced = is_equally_spaced
        self.is_empty = is_empty
        self.has_nans = has_nans
        self.n_features = n_features
        self.feature_names = feature_names

        super().__init__()
