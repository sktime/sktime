"""Test functions for load_hierarchical_sales_toydata."""

__author__ = ["fkiraly"]

from sktime.datatypes import check_raise


def test_load_hierarchical_sales_toydata():
    """Test that load_hierarchical_sales_toydata runs and returns expected format."""
    from sktime.datasets import load_hierarchical_sales_toydata

    df = load_hierarchical_sales_toydata()

    check_raise(
        df,
        "pd_multiindex_hier",
        var_name="return of load_hierarchical_sales_toydata",
    )
