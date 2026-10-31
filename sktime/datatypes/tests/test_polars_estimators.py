"""Integration tests for polars input and output in estimators and pipelines."""

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import check_is_mtype, convert
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.dependencies import _check_soft_dependencies

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
HIERARCHY_LEVELS = {"Series": (), "Panel": (3,), "Hierarchical": (2, 2)}
SCITYPES = list(POLARS_MTYPES)

SKIP_REASON = "Test only if datatypes or base classes changed, and if polars present"
skip_if_unchanged = pytest.mark.skipif(
    not run_test_module_changed(
        [
            "sktime.datatypes",
            "sktime.forecasting.base",
            "sktime.forecasting.compose",
            "sktime.transformations.base",
        ]
    )
    or not _check_soft_dependencies("polars", severity="none"),
    reason=SKIP_REASON,
)


def _make_polars(scitype):
    """Return polars data of the given scitype, with positive values."""
    X = _make_hierarchical(
        hierarchy_levels=HIERARCHY_LEVELS[scitype],
        n_columns=1,
        min_timepoints=12,
        max_timepoints=12,
        all_positive=True,
    )
    if scitype == "Series":
        X = X.droplevel(0) if X.index.nlevels > 1 else X
    return convert(X, PANDAS_MTYPES[scitype], POLARS_MTYPES[scitype], scitype)


def _to_pandas(X, scitype):
    return convert(X, POLARS_MTYPES[scitype], PANDAS_MTYPES[scitype], scitype)


def _assert_polars_mtype(X, scitype):
    import polars as pl

    assert isinstance(X, pl.DataFrame)
    assert check_is_mtype(X, POLARS_MTYPES[scitype], scitype, return_metadata=False)


def _assert_same_values(X_pl, X_pd, scitype):
    """Assert polars result has the same values and index as the pandas result."""
    X_pl = _to_pandas(X_pl, scitype)
    assert (X_pl.index == X_pd.index).all()
    assert np.allclose(X_pl.values, X_pd.values)


@skip_if_unchanged
@pytest.mark.parametrize("scitype", SCITYPES)
def test_polars_forecaster(scitype):
    """Test a forecaster on polars input returns polars of the same scitype."""
    from sktime.forecasting.naive import NaiveForecaster

    y = _make_polars(scitype)
    fh = [1, 2, 3]

    y_pred = NaiveForecaster().fit(y, fh=fh).predict()
    y_pred_pd = NaiveForecaster().fit(_to_pandas(y, scitype), fh=fh).predict()

    _assert_polars_mtype(y_pred, scitype)
    _assert_same_values(y_pred, y_pred_pd, scitype)


@skip_if_unchanged
@pytest.mark.parametrize("scitype", SCITYPES)
def test_polars_series_to_series_transformer(scitype):
    """Test a series-to-series transformer on polars input, and its inverse."""
    from sktime.transformations.series.exponent import ExponentTransformer

    X = _make_polars(scitype)

    t = ExponentTransformer().fit(X)
    Xt = t.transform(X)
    Xt_pd = ExponentTransformer().fit_transform(_to_pandas(X, scitype))

    _assert_polars_mtype(Xt, scitype)
    _assert_same_values(Xt, Xt_pd, scitype)

    Xit = t.inverse_transform(Xt)
    _assert_polars_mtype(Xit, scitype)
    _assert_same_values(Xit, _to_pandas(X, scitype), scitype)


@skip_if_unchanged
@pytest.mark.parametrize("scitype", SCITYPES)
def test_polars_series_to_primitives_transformer(scitype):
    """Test a series-to-primitives transformer on polars input.

    Primitives output is a polars_eager_table, with the same values as for pandas.
    The index of the pandas output is kept as ``__index__`` columns.
    """
    import polars as pl

    from sktime.transformations.series.summarize import SummaryTransformer

    X = _make_polars(scitype)

    Xt = SummaryTransformer().fit_transform(X)
    Xt_pd = SummaryTransformer().fit_transform(_to_pandas(X, scitype))

    assert isinstance(Xt, pl.DataFrame)
    assert check_is_mtype(Xt, "polars_eager_table", "Table", return_metadata=False)

    index_cols = [x for x in Xt.columns if x.startswith("__index__")]
    default_index = Xt_pd.index.equals(pd.RangeIndex(len(Xt_pd)))
    assert len(index_cols) == (0 if default_index else Xt_pd.index.nlevels)
    assert list(Xt.drop(index_cols).columns) == [str(x) for x in Xt_pd.columns]
    assert np.allclose(Xt.drop(index_cols).to_numpy(), Xt_pd.to_numpy())


@skip_if_unchanged
@pytest.mark.parametrize("scitype", SCITYPES)
def test_polars_forecasting_pipeline(scitype):
    """Test a forecasting pipeline with a transformer on polars input."""
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.transformations.series.exponent import ExponentTransformer

    def _pipeline():
        return TransformedTargetForecaster(
            [ExponentTransformer(), NaiveForecaster(strategy="mean")]
        )

    y = _make_polars(scitype)
    fh = [1, 2, 3]

    y_pred = _pipeline().fit(y, fh=fh).predict()
    y_pred_pd = _pipeline().fit(_to_pandas(y, scitype), fh=fh).predict()

    _assert_polars_mtype(y_pred, scitype)
    _assert_same_values(y_pred, y_pred_pd, scitype)
