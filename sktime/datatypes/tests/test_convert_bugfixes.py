"""Tests for specific bugfixes to conversion logic."""

__author__ = ["fkiraly", "ericjb"]

import pytest

from sktime.datasets import load_airline
from sktime.datatypes._series._convert import convert_MvS_to_UvS_as_Series
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_multiindex_to_df_list_large_level_values():
    """Tests for failure condition in bug #4668.

    Conversion from pd-multiindex to df-list would fail if the
    first MultiIndex level (level index 0) had strictly more levels
    than unique values in it, this can occur post subsetting.
    """
    from sktime.datasets import load_osuleaf
    from sktime.datatypes import convert_to

    X, _ = load_osuleaf(return_type="pd-multiindex")
    X1 = X.loc[:3]

    convert_to(X1, "df-list")


@pytest.mark.xfail(reason="Failing test for bug #7928, to be fixed")
@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_MvS_to_UvS_as_Series():
    """Checks that column name in MvS is preserved as attr name in UvS"""
    y = load_airline()
    z = y.to_frame()
    w = convert_MvS_to_UvS_as_Series(z)

    assert y.name == w.name


def test_proba_base_converters():
    """Test BaseConverter interface for Proba converters."""
    from sktime.datatypes._examples import get_examples
    from sktime.datatypes._proba._convert import (
        ProbaPredIntervalToQuantiles,
        ProbaPredQuantilesToInterval,
    )
    from sktime.utils.deep_equals import deep_equals

    c1 = ProbaPredIntervalToQuantiles()
    assert c1.get_tag("mtype_from") == "pred_interval"
    assert c1.get_tag("mtype_to") == "pred_quantiles"
    assert c1.get_conversions() == [("pred_interval", "pred_quantiles")]

    c2 = ProbaPredQuantilesToInterval()
    assert c2.get_tag("mtype_from") == "pred_quantiles"
    assert c2.get_tag("mtype_to") == "pred_interval"
    assert c2.get_conversions() == [("pred_quantiles", "pred_interval")]

    ex_interval = get_examples(mtype="pred_interval", as_scitype="Proba")[0]
    ex_quantiles = get_examples(mtype="pred_quantiles", as_scitype="Proba")[0]

    converted_q = c1(ex_interval)
    assert deep_equals(converted_q, ex_quantiles)

    converted_i = c2(ex_quantiles)
    assert deep_equals(converted_i, ex_interval)
