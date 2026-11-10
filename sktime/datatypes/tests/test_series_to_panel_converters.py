"""Testing panel converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import check_is_mtype
from sktime.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
    convert_to_scitype,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel
from sktime.utils._testing.series import _make_series


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
