# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseDatatype


class ScitypeHierarchical(BaseDatatype):
    r"""Hierarchical data type. Represents a hierarchical collection of time series.

    The ``Hierarchical`` data type is an abstract data type (= :term:`scitype`).

    A ``Hierarchical`` represents a hierarchically indexed collection of
    many single monotonously indexed sequences,
    that is, an indexed collection of objects that follow the ``Series`` data type,
    where the index is hierarchical (multi-level).

    This including the sub-case of what is commonly called a "hierarchical time series",
    if the index is interpreted as time.

    Formally, an abstract ``Hierarchical`` object that
    contains :math:`N` time series has:

    * an index :math:`s_1, \dots, s_N`, where each :math:`s_i` is an :math:`H`-tuple
      of integer or a categorical types, representing the series identifier, also called
      "hierarchical index"
    * individual time series :math:`S_i` for :math:`i = 1, \dots, N`,
      where each :math:`S_i` is an object of abstract ``Series`` type
    * above, the domain of the hierarchical index is a
      fixed Cartesian product of domains,
      :math:`\mathcal{S} = \mathcal{S}_1 \times \dots \times \mathcal{S}_H`
      where each :math:`\mathcal{S}_h` is an integer or categorical type domain.

    To be considered of ``Hierarchical`` type, there must be at least two hierarchy
    levels, i.e., :math:`H \geq 2`. If :math:`H = 1`, the data is considered of
    ``Panel`` type (to avoid duplicate typing).

    The object :math:`S_i` is interpreted as the time series (or sequence) at
    the instance index :math:`s_i`.

    The indices :math:`s_i` are assumed distinct, but not necessarily ordered.

    The domains :math:`\mathcal{S}_h` are interpreted as hierarchy levels.

    Concrete types implementing the ``Hierarchical`` data type must specify:

    * hierarchy: how the hierarchy levels are represented
    * hierarchy names: optional, names of the hierarchy levels
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
        True iff all series in the hierarchical collection are equally spaced
    is_equal_length: bool
        True iff all series in the hierarchical collection are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the hierarchical collection.
    is_one_panel: bool
        True iff there is only one flat panel in the hierarchical collection,
        i.e., the collection has a flat hierarchy, plus additional hierarchy levels
        that have only a single value.
    has_nans: bool
        True iff the hierarchical series contains NaN values
    n_instances: int
        number of instances, i.e., individual time series, in the hierarchical
        collection. Formally, the number of unique hierarchy indices :math:`s_i`,
        namely :math:`N`, in the definition above.
    n_panels: int
        number of flat panels in the hierarchical collection. If the indices are
        :math:`H`-tuples, this is the number of unique values of the
        :math:`(H-1)`-tuples obtained by removing the last entry of each :math:`s_i`.
    n_features: int
        number of variables in the hierarchical series
    feature_names: list of int or object
        names of variables in the hierarchical series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the hierarchical series,
        following the data frame interface protocol.
        In same order as ``feature_names``.
    feature_kind: list of str
        list of feature-kind strings for each feature in the hierarchical series,
        coerced to ``"FLOAT"`` or ``"CATEGORICAL"`` type string.
        In same order as ``feature_names``.
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
