"""Tests for sklearn dataframe coercion."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.sklearn._adapt_df import prep_skl_df


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.sklearn"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("copy_df", [True, False])
def test_prep_skl_df_coercion(copy_df):
    """Test that prep_skl_df behaves correctly on the coercion case."""
    mixed_example = pd.DataFrame({0: [1, 2, 3], "b": [1, 2, 3]})

    res = prep_skl_df(mixed_example, copy_df=copy_df)

    assert np.all(res.columns == ["0", "b"])

    if not copy_df:
        assert res is mixed_example


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.sklearn"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("copy_df", [True, False])
def test_prep_skl_df_non_coercion(copy_df):
    """Test that prep_skl_df behaves correctly on the non-coercion case."""
    mixed_example = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    res = prep_skl_df(mixed_example, copy_df=copy_df)

    assert np.all(res.columns == ["a", "b"])
    assert res is mixed_example
