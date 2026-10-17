"""Testing panel converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
)
from sktime.datatypes._series_as_panel._convert import (
    DfListToPdDataFrame,
    HierarchicalToPanel,
    HierarchicalToSeries,
    Numpy3DToNumpy,
    NumpyToNumpy3D,
    PanelToHierarchical,
    PdDataFrameToDfList,
    PdMultiIndexToPdDataFrame,
    PdSeriesToDfList,
    SeriesToHierarchical,
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


CONVERTERS = [
    (PdSeriesToDfList, "Series", "Panel"),
    (PdDataFrameToDfList, "Series", "Panel"),
    (NumpyToNumpy3D, "Series", "Panel"),
    (DfListToPdDataFrame, "Panel", "Series"),
    (PdMultiIndexToPdDataFrame, "Panel", "Series"),
    (Numpy3DToNumpy, "Panel", "Series"),
    (SeriesToHierarchical, "Series", "Hierarchical"),
    (HierarchicalToSeries, "Hierarchical", "Series"),
    (PanelToHierarchical, "Panel", "Hierarchical"),
    (HierarchicalToPanel, "Hierarchical", "Panel"),
]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("converter_cls, scitype_from, scitype_to", CONVERTERS)
def test_converter_class_tags(converter_cls, scitype_from, scitype_to):
    """Test that each converter class declares the conversion it implements."""
    converter = converter_cls()
    assert converter.get_tag("object_type") == "converter"
    assert converter.get_tag("scitype_from") == scitype_from
    assert converter.get_tag("scitype_to") == scitype_to
    assert converter.get_tag("mtype_to") is not None
    mtype_pair = (converter.get_tag("mtype_from"), converter.get_tag("mtype_to"))
    assert converter_cls.get_conversions() == [mtype_pair]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_converter_class_roundtrip():
    """Test Series to Panel and back via converter classes."""
    X_series = _make_series(n_columns=2, return_mtype="pd.DataFrame")

    X_panel = PdDataFrameToDfList()(X_series)
    assert isinstance(X_panel, list) and len(X_panel) == 1

    X_back = DfListToPdDataFrame()(X_panel)
    assert isinstance(X_back, pd.DataFrame)
    assert (X_back.values == X_series.values).all()
