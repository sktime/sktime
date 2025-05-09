# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypeHierarchical(BaseDatatype):
    """Hierarchical data type. Represents a hierarchical collection of time series.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the hierarchical collection.
    is_one_panel: bool
        True iff there is only one flat panel in the hierarchical collection,
        i.e., the collection has a flat hierarchy, plus additional hierarchy levels
        that have only a single value.
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the hierarchical collection
    n_panels: int
        number of flat panels in the hierarchical collection
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Hierarchical",
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
        is_equal_length=None,
        is_empty=None,
        is_one_series=None,
        is_one_panel=None,
        has_nans=None,
        n_instances=None,
        n_panels=None,
        n_features=None,
        feature_names=None,
        dtypekind_dfip=None,
        feature_kind=None,
    ):
        self.is_univariate = is_univariate
        self.is_equally_spaced = is_equally_spaced
        self.is_equal_length = is_equal_length
        self.is_empty = is_empty
        self.is_one_series = is_one_series
        self.is_one_panel = is_one_panel
        self.has_nans = has_nans
        self.n_instances = n_instances
        self.n_panels = n_panels
        self.n_features = n_features
        self.feature_names = feature_names
        self.dtypekind_dfip = dtypekind_dfip
        self.feature_kind = feature_kind

        super().__init__()
