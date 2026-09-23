"""Testing panel converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
)
from sktime.tests.test_switch import run_test_module_changed
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
def test_convert_polars_series_to_panel():
    """Test output format of series-to-panel for polars DataFrame input."""
    from sktime.utils.dependencies import _check_soft_dependencies

    if not _check_soft_dependencies("polars", severity="none"):
        pytest.skip("polars not available")

    import polars as pl

    from sktime.datatypes._series_as_panel import convert_Series_to_Panel

    df_series = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    panel = convert_Series_to_Panel(df_series)

    assert isinstance(panel, pl.DataFrame)
    panel_with_mtype, mtype = convert_Series_to_Panel(df_series, return_to_mtype=True)
    assert mtype == "polars_panel"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_polars_panel_to_series():
    """Test output format of panel-to-series for polars DataFrame input."""
    from sktime.utils.dependencies import _check_soft_dependencies

    if not _check_soft_dependencies("polars", severity="none"):
        pytest.skip("polars not available")

    import polars as pl

    from sktime.datatypes._series_as_panel import (
        convert_Panel_to_Series,
        convert_Series_to_Panel,
    )

    df_series = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    panel = convert_Series_to_Panel(df_series)

    recovered_series = convert_Panel_to_Series(panel)
    assert isinstance(recovered_series, pl.DataFrame)
    assert recovered_series.shape == df_series.shape

    recovered, mtype = convert_Panel_to_Series(panel, return_to_mtype=True)
    assert mtype == "polars_series"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_series_panel_unsupported_type_raises():
    """Test TypeError is raised on unsupported mtypes."""
    from sktime.datatypes._series_as_panel import (
        convert_Panel_to_Series,
        convert_Series_to_Panel,
    )

    with pytest.raises(TypeError):
        convert_Series_to_Panel("invalid_series")

    with pytest.raises(TypeError):
        convert_Panel_to_Series("invalid_panel")
