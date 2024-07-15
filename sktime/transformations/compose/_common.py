"""Common components and constants for transformer compositors."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.utils.sklearn import is_sklearn_transformer

# mtypes for Series, Panel, Hierarchical,
# with exception of some ambiguous and discouraged mtypes
CORE_MTYPES = [
    "pd.DataFrame",
    "np.ndarray",
    "pd.Series",
    "pd-multiindex",
    "df-list",
    "nested_univ",
    "numpy3D",
    "pd_multiindex_hier",
]


def _coerce_to_sktime(other):
    """Check and format inputs to dunders for compose."""
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor

    # if sklearn transformer, adapt to sktime transformer first
    if is_sklearn_transformer(other):
        return TabularToSeriesAdaptor(other)

    return other
