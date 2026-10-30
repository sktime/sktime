"""Scitype conversions for polars based mtypes.

Converts between Series, Panel and Hierarchical by adding or removing
the ``__index__`` columns of the polars frame, without conversion to pandas.

The result is the same as converting to pandas, applying the pandas based
conversion, and converting back to polars, including names of index columns.
"""

__author__ = ["PragnyaKhandelwal"]

import sys

POLARS_MTYPES = {
    "Series": "pl.DataFrame",
    "Panel": "polars_panel",
    "Hierarchical": "polars_hierarchical",
}

_PREFIX = "__index__"


def is_polars_frame(obj):
    """Check whether obj is a polars DataFrame, without importing polars."""
    pl = sys.modules.get("polars")
    return pl is not None and isinstance(obj, pl.DataFrame)


def _columns(obj):
    return obj.collect_schema().names()


def _index_cols(obj):
    return [x for x in _columns(obj) if x.startswith(_PREFIX)]


def _ensure_time_index(obj):
    """Add an integer time index column if obj has no index columns."""
    import polars as pl

    if len(_index_cols(obj)) > 0:
        return obj
    time_col = pl.int_range(pl.len(), dtype=pl.Int64).alias(f"{_PREFIX}0")
    return obj.select([time_col, pl.all()])


def _add_index_cols(obj, names):
    """Add constant zero index columns with names, as outermost levels."""
    import polars as pl

    obj = _ensure_time_index(obj)
    new_cols = [pl.lit(0, dtype=pl.Int64).alias(f"{_PREFIX}{x}") for x in names]
    return obj.select(new_cols + [pl.all()])


def _keep_index_cols(obj, n_keep):
    """Remove all but the n_keep innermost index columns."""
    index_cols = _index_cols(obj)
    return obj.drop(index_cols[: len(index_cols) - n_keep])


def _renumber_unnamed(obj):
    """Rename unnamed index columns to their new position, as pandas does."""
    rename = {}
    for i, col in enumerate(_index_cols(obj)):
        name = col[len(_PREFIX) :]
        if name.isdigit() and int(name) != i:
            rename[col] = f"{_PREFIX}{i}"
    if rename:
        obj = obj.rename(rename)
    return obj


def _series_to_panel(obj):
    obj = _ensure_time_index(obj)
    obj = obj.rename({_index_cols(obj)[-1]: f"{_PREFIX}timepoints"})
    return _add_index_cols(obj, ["instances"])


def _panel_to_series(obj):
    return _keep_index_cols(obj, 1)


def _series_to_hierarchical(obj):
    return _add_index_cols(obj, ["__level1", "__level2"])


def _hierarchical_to_series(obj):
    return _keep_index_cols(obj, 1)


def _panel_to_hierarchical(obj):
    return _add_index_cols(obj, ["__level2"])


def _hierarchical_to_panel(obj):
    return _keep_index_cols(obj, 2)


_POLARS_CONVERTERS = {
    ("Series", "Panel"): _series_to_panel,
    ("Panel", "Series"): _panel_to_series,
    ("Series", "Hierarchical"): _series_to_hierarchical,
    ("Hierarchical", "Series"): _hierarchical_to_series,
    ("Panel", "Hierarchical"): _panel_to_hierarchical,
    ("Hierarchical", "Panel"): _hierarchical_to_panel,
}


def convert_polars(obj, scitype_from, scitype_to, return_to_mtype=False):
    """Convert polars frame obj from scitype_from to scitype_to.

    Parameters
    ----------
    obj : polars.DataFrame, of polars mtype of scitype_from
    scitype_from : str, one of "Series", "Panel", "Hierarchical"
    scitype_to : str, one of "Series", "Panel", "Hierarchical"
    return_to_mtype : bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    polars.DataFrame, of polars mtype of scitype_to
    """
    obj = _POLARS_CONVERTERS[(scitype_from, scitype_to)](obj)
    obj = _renumber_unnamed(obj)
    if return_to_mtype:
        return obj, POLARS_MTYPES[scitype_to]
    return obj
