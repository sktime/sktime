# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypePanel(BaseDatatype):
    r"""Panel data type. Represents a panel (flat collection) of time series.

    The ``Panel`` data type is an abstract data type (= :term:`scitype`).

    It represents an indexed collection of many single monotonously indexed sequences,
    that is, an indexed collection of objects that follow the ``Series`` data type.

    This including the sub-case of a panel of "time series" if the index is interpreted
    as time.

    Formally, an abstract ``Panel`` object that contains :math:`N` time series has:

    * an index :math:`s_1, \dots, s_N`, with :math:`s_i` being an integer or
      a categorical type, representing the series identifier, also called
      "instance index" or "case index"
    * individual time series :math:`S_i` for :math:`i = 1, \dots, N`, each being an
      object of abstract ``Series`` type

    The object :math:`S_i` is interpreted as the time series (or sequence) at
    the instance index :math:`s_i`.

    The indices :math:`s_i` are assumed distinct, but not necessarily ordered.

    Concrete types implementing the ``Panel`` data type must specify:

    * instances: how the different instances are represented
    * instance names: optional, names of the instance dimensions
    * variables: how the dimensions of the individual time series are represented
    * variable names: optional, names of the variable dimensions
    * time points: how time points are represented in the individual time series
    * time index: how the time index is represented

    Concrete implementations may implement only sub-cases of the full abstract type.

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
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
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
        "scitype": "Panel",
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
        has_nans=None,
        n_instances=None,
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
        self.has_nans = has_nans
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_names = feature_names
        self.dtypekind_dfip = dtypekind_dfip
        self.feature_kind = feature_kind

        super().__init__()
