# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypeTable(BaseDatatype):
    r"""Data Frame or Table data type.

    The ``Table`` data type is an abstract data type (= :term:`scitype`).

    It represents a row- and column-indexed 2D table of data, commonly
    referred to as "data frame".

    Formally, an abstract ``Table`` object has:

    * an index :math:`i_1, \dots, i_T`, with :math:`i_i` being any hashable type
    * values :math:`x_1, \dots, x_T`, with :math:`x_i`, taking values in
      an abstract typed data frame row domain :math:`\mathcal{Y}`,
      i.e., vectors with entries being numbers
      (float, integer) or categorical, always the same type at the same entry

    The value :math:`x_i` is interpreted to be an "observation" or "instance"
    at index :math:`i_i`.

    The indices :math:`i_i` are assumed distinct, but not necessarily ordered.

    Concrete types implementing the ``Table`` data type must specify:

    * features: how the dimensions of :math:`\mathcal{Y}` are represented
    * feature names: optional, names of the column dimensions
    * instances: how the value "observed at" an index is represented
    * instance index: how the instance index is represented

    Concrete implementations may implement only sub-cases of the full abstract type.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the table,
        following the data frame interface protocol.
        In same order as ``feature_names``.
    feature_kind: list of str
        list of feature-kind strings for each feature in the table,
        coerced to ``"FLOAT"`` or ``"CATEGORICAL"`` type string.
        In same order as ``feature_names``.
    """

    _tags = {
        "scitype": "Table",
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(
        self,
        is_univariate=None,
        is_empty=None,
        has_nans=None,
        n_instances=None,
        n_features=None,
        feature_names=None,
        dtypekind_dfip=None,
        feature_kind=None,
    ):
        self.is_univariate = is_univariate
        self.is_empty = is_empty
        self.has_nans = has_nans
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_names = feature_names
        self.dtypekind_dfip = dtypekind_dfip
        self.feature_kind = feature_kind

        super().__init__()
