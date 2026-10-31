"""Testing panel converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import check_is_mtype, convert, convert_to, get_examples
from sktime.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
    convert_to_scitype,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel
from sktime.utils._testing.series import _make_series
from sktime.utils.deep_equals import deep_equals
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_numpy_series_to_panel():
    """Test output format of series-to-panel for numpy type input."""
    X_series = _make_series(n_columns=2, return_mtype="np.ndarray")
    n_time, n_var = X_series.shape

    X_panel = convert_Series_to_Panel(X_series)

    assert isinstance(X_panel, np.ndarray)
    assert X_panel.ndim == 3
    assert X_panel.shape == (1, n_var, n_time)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_numpy_panel_to_series():
    """Test output format of panel-to-series for numpy type input."""
    X_panel = _make_panel(n_instances=1, n_columns=2, return_mtype="numpy3D")
    _, n_var, n_time = X_panel.shape

    X_series = convert_Panel_to_Series(X_panel)

    assert isinstance(X_series, np.ndarray)
    assert X_series.ndim == 2
    assert X_series.shape == (n_time, n_var)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_df_series_to_panel():
    """Test output format of series-to-panel for dataframe type input."""
    X_series = _make_series(n_columns=2, return_mtype="pd.DataFrame")

    X_panel = convert_Series_to_Panel(X_series)

    assert isinstance(X_panel, list)
    assert isinstance(X_panel[0], pd.DataFrame)
    assert X_panel[0].equals(X_series)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_df_panel_to_series():
    """Test output format of panel-to-series for dataframe type input."""
    X_panel = _make_panel(n_instances=1, n_columns=2, return_mtype="pd-multiindex")

    X_series = convert_Panel_to_Series(X_panel)

    assert isinstance(X_series, pd.DataFrame)
    assert len(X_series) == len(X_panel)
    assert (X_series.values == X_panel.values).all()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("hierarchy_levels", [(1, 3), (1, 1, 3)])
def test_convert_hierarchical_to_panel(hierarchy_levels):
    """Test hierarchical-to-panel keeps the instance and time index levels."""
    X_hier = _make_hierarchical(hierarchy_levels=hierarchy_levels, n_columns=1)

    X_panel, mtype = convert_to_scitype(
        X_hier, to_scitype="Panel", from_scitype="Hierarchical", return_to_mtype=True
    )

    assert mtype == "pd-multiindex"
    assert check_is_mtype(X_panel, "pd-multiindex", return_metadata=False)
    assert X_panel.index.nlevels == 2
    assert X_panel.index.equals(
        X_hier.index.droplevel(list(range(X_hier.index.nlevels - 2)))
    )
    assert (X_panel.values == X_hier.values).all()


POLARS_MTYPES = {
    "Series": "pl.DataFrame",
    "Panel": "polars_panel",
    "Hierarchical": "polars_hierarchical",
}
PANDAS_MTYPES = {
    "Series": "pd.DataFrame",
    "Panel": "pd-multiindex",
    "Hierarchical": "pd_multiindex_hier",
}


def _polars_fixture(scitype):
    """Return a single-instance polars fixture of the given scitype."""
    if scitype == "Series":
        return get_examples(mtype="pl.DataFrame", as_scitype="Series")[0]
    if scitype == "Panel":
        X = _make_panel(n_instances=1, n_columns=2, return_mtype="pd-multiindex")
        return convert(X, "pd-multiindex", "polars_panel", "Panel")
    X = _make_hierarchical(hierarchy_levels=(1, 1), n_columns=2)
    return convert(X, "pd_multiindex_hier", "polars_hierarchical", "Hierarchical")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes")
    or not _check_soft_dependencies("polars", severity="none"),
    reason="Test only if sktime.datatypes changed, and if polars is present",
)
@pytest.mark.parametrize(
    "scitype_from, scitype_to",
    [
        ("Series", "Panel"),
        ("Panel", "Series"),
        ("Series", "Hierarchical"),
        ("Hierarchical", "Series"),
        ("Panel", "Hierarchical"),
        ("Hierarchical", "Panel"),
    ],
)
def test_convert_to_scitype_polars(scitype_from, scitype_to):
    """Test polars inputs are converted to the polars mtype of the target scitype.

    The result must be a valid polars mtype, and equal to the pandas conversion
    of the same data.
    """
    X = _polars_fixture(scitype_from)

    Xt, mtype = convert_to_scitype(
        X, to_scitype=scitype_to, from_scitype=scitype_from, return_to_mtype=True
    )

    assert mtype == POLARS_MTYPES[scitype_to]
    assert check_is_mtype(Xt, mtype, scitype_to, return_metadata=False)

    X_pd = convert(
        X, POLARS_MTYPES[scitype_from], PANDAS_MTYPES[scitype_from], scitype_from
    )
    Xt_pd_expected = convert_to_scitype(
        X_pd, to_scitype=scitype_to, from_scitype=scitype_from
    )
    Xt_pd_expected = convert_to(Xt_pd_expected, PANDAS_MTYPES[scitype_to])
    Xt_pd = convert(Xt, mtype, PANDAS_MTYPES[scitype_to], scitype_to)
    assert deep_equals(Xt_pd, Xt_pd_expected)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes")
    or not _check_soft_dependencies("polars", severity="none"),
    reason="Test only if sktime.datatypes changed, and if polars is present",
)
def test_convert_hierarchical_to_panel_polars_deep():
    """Test polars hierarchical to panel for more than 3 index levels."""
    X = _make_hierarchical(hierarchy_levels=(1, 1, 3), n_columns=1)
    X = convert(X, "pd_multiindex_hier", "polars_hierarchical", "Hierarchical")

    Xt = convert_to_scitype(X, to_scitype="Panel", from_scitype="Hierarchical")

    assert check_is_mtype(Xt, "polars_panel", "Panel", return_metadata=False)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("fun", [convert_Series_to_Panel, convert_Panel_to_Series])
def test_unsupported_type_raises_type_error(fun):
    """Test unsupported input types raise TypeError when the mtype is requested."""
    with pytest.raises(TypeError, match="obj must be of mtype"):
        fun(object(), return_to_mtype=True)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes")
    or not _check_soft_dependencies("polars", severity="none"),
    reason="Test only if sktime.datatypes changed, and if polars is present",
)
@pytest.mark.parametrize(
    "scitype_from, scitype_to",
    [
        ("Series", "Panel"),
        ("Panel", "Series"),
        ("Series", "Hierarchical"),
        ("Hierarchical", "Series"),
        ("Panel", "Hierarchical"),
        ("Hierarchical", "Panel"),
    ],
)
def test_convert_to_scitype_polars_no_pandas(scitype_from, scitype_to, monkeypatch):
    """Test polars conversions manipulate the index columns, without pandas."""
    import polars as pl

    X = _polars_fixture(scitype_from)

    def _raise(*args, **kwargs):
        raise AssertionError("polars frame was converted to pandas")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", _raise)

    Xt = convert_to_scitype(X, to_scitype=scitype_to, from_scitype=scitype_from)
    assert isinstance(Xt, pl.DataFrame)
