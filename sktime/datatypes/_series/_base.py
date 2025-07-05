# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypeSeries(BaseDatatype):
    r"""Series data type. Represents a single time series.

    The ``Series`` data type is an abstract data type (= :term:`scitype`).

    It represents a single monotonously indexed sequence, including the sub-case of
    "time series" if the index is interpreted as time.

    Formally, an abstract ``Series`` object has:

    * an index :math:`t_1, \dots, t_T`, with :math:`t_i` being an integer or
      an orderable datetime-like type, representing the time point (or index)
    * values :math:`y_1, \dots, y_T`, with :math:`y_i`, taking values in
      an abstract typed data frame row domain :math:`\mathcal{Y}`,
      i.e., vectors with entries being numbers
      (float, integer) or categorical, always the same type at the same entry

    The value :math:`y_i` is interpreted to be "observed at time point" :math:`t_i`.

    The indices are assumed distinct and ordered, i.e., :math:`t_{i-1} < t_i`
    for all :math:`i`.

    Concrete types implementing the ``Series`` data type must specify:

    * variables: how the dimensions of :math:`\mathcal{Y}` are represented
    * variable names: optional, names of the variable dimensions
    * time points: how the value "observed at" a given time point is represented
    * time index: how the time index is represented

    Concrete implementations may implement only sub-cases of the full abstract type.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
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
        dtypekind_dfip=None,
        feature_kind=None,
    ):
        self.is_univariate = is_univariate
        self.is_equally_spaced = is_equally_spaced
        self.is_empty = is_empty
        self.has_nans = has_nans
        self.n_features = n_features
        self.feature_names = feature_names
        self.dtypekind_dfip = dtypekind_dfip
        self.feature_kind = feature_kind

        super().__init__()
