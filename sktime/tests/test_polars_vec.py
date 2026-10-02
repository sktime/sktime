# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for native Polars support in VectorizedDF."""

import pytest
import pandas as pd

pl = pytest.importorskip("polars")  # entire file skipped if polars not installed

from sktime.datatypes._vectorize import VectorizedDF
from sktime.datatypes._adapter.polars import convert_pandas_to_polars


def _make_panel():
    """Minimal valid sktime Panel in pd-multiindex format."""
    return pd.DataFrame(
        {
            "var1": [1.1, 1.2, 2.1, 2.2],
            "var2": [3.1, 3.2, 4.1, 4.2],
        },
        index=pd.MultiIndex.from_tuples(
            [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
            names=["instances", "timepoints"],
        ),
    )


# ===========================================================================
# Eager pl.DataFrame tests
# ===========================================================================


def test_eager_flags():
    """Eager pl.DataFrame: is_polars=True, is_lazy=False."""
    pl_df = convert_pandas_to_polars(_make_panel())
    vec = VectorizedDF(pl_df, is_scitype="Panel", iterate_as="Series")

    assert vec.is_polars is True
    assert vec.is_lazy is False


def test_eager_iteration_count():
    """Eager pl.DataFrame: should yield one slice per instance."""
    pl_df = convert_pandas_to_polars(_make_panel())
    vec = VectorizedDF(pl_df, is_scitype="Panel", iterate_as="Series")

    result = list(vec)
    assert len(result) == 2


def test_eager_iteration_type():
    """Eager pl.DataFrame: each yielded slice must be a pl.DataFrame."""
    pl_df = convert_pandas_to_polars(_make_panel())
    vec = VectorizedDF(pl_df, is_scitype="Panel", iterate_as="Series")

    for item in vec:
        assert isinstance(item, pl.DataFrame)


def test_eager_reconstruct_type():
    """Eager pl.DataFrame: reconstruct returns pl.DataFrame."""
    pl_df = convert_pandas_to_polars(_make_panel())
    vec = VectorizedDF(pl_df, is_scitype="Panel", iterate_as="Series")

    slices = list(vec)
    recon = vec.reconstruct(slices)

    assert isinstance(recon, pl.DataFrame)


def test_eager_reconstruct_row_count():
    """Eager pl.DataFrame: reconstructed frame has all original rows."""
    pl_df = convert_pandas_to_polars(_make_panel())
    vec = VectorizedDF(pl_df, is_scitype="Panel", iterate_as="Series")

    slices = list(vec)
    recon = vec.reconstruct(slices)

    assert recon.height == 4


# ===========================================================================
# Lazy pl.LazyFrame tests
# ===========================================================================


def test_lazy_flags():
    """Lazy pl.LazyFrame: is_polars=True, is_lazy=True."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    assert vec.is_polars is True
    assert vec.is_lazy is True


def test_lazy_no_collect_at_init():
    """Lazy path: row_ix is None at init, _lazy_group_cols is set."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    row_ix, _ = vec.get_iter_indices()

    # row_ix must be None — keys are deferred, not collected at init
    assert row_ix is None

    # group cols must be stored for deferred iteration
    assert hasattr(vec, "_lazy_group_cols")
    assert len(vec._lazy_group_cols) > 0


def test_lazy_len_raises():
    """Lazy path: len() must raise TypeError, not return a wrong value."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    with pytest.raises(TypeError, match="lazy"):
        len(vec)


def test_lazy_iteration_count():
    """Lazy pl.LazyFrame: should yield one slice per instance."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    result = list(vec)
    assert len(result) == 2


def test_lazy_iteration_type():
    """Lazy pl.LazyFrame: each yielded slice must be a pl.LazyFrame."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    for item in vec:
        assert isinstance(item, pl.LazyFrame)


def test_lazy_reconstruct_type():
    """Lazy pl.LazyFrame: reconstruct returns pl.LazyFrame for convert_back=True."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    slices = list(vec)
    recon = vec.reconstruct(slices, convert_back=True)

    assert isinstance(recon, pl.LazyFrame)


def test_lazy_reconstruct_row_count():
    """Lazy pl.LazyFrame: reconstructed frame has all original rows."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    slices = list(vec)
    recon = vec.reconstruct(slices, convert_back=True)

    assert recon.collect().height == 4


def test_lazy_key_order_matches_slices():
    """Lazy path: _lazy_yielded_keys are populated after iteration."""
    lazy_df = convert_pandas_to_polars(_make_panel()).lazy()
    vec = VectorizedDF(lazy_df, is_scitype="Panel", iterate_as="Series")

    _ = list(vec)  # trigger iteration to populate cache

    assert len(vec._lazy_yielded_keys) == 2


# ===========================================================================
# Pandas regression tests — must be completely unaffected
# ===========================================================================


def test_pandas_flags():
    """Pandas path: is_polars=False, is_lazy=False."""
    vec = VectorizedDF(_make_panel(), is_scitype="Panel", iterate_as="Series")

    assert vec.is_polars is False
    assert vec.is_lazy is False


def test_pandas_iteration_count():
    """Pandas path: iteration count unchanged."""
    vec = VectorizedDF(_make_panel(), is_scitype="Panel", iterate_as="Series")

    assert len(list(vec)) == 2


def test_pandas_iteration_type():
    """Pandas path: yielded slices are still pd.DataFrame."""
    vec = VectorizedDF(_make_panel(), is_scitype="Panel", iterate_as="Series")

    for item in vec:
        assert isinstance(item, pd.DataFrame)


def test_pandas_reconstruct_type():
    """Pandas path: reconstruct still returns pd.DataFrame."""
    vec = VectorizedDF(_make_panel(), is_scitype="Panel", iterate_as="Series")
    slices = list(vec)
    recon = vec.reconstruct(slices)

    assert isinstance(recon, pd.DataFrame)


def test_pandas_len_works():
    """Pandas path: len() still works correctly."""
    vec = VectorizedDF(_make_panel(), is_scitype="Panel", iterate_as="Series")

    assert len(vec) == 2