"""Column concatenator test code."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.colconcat import ColumnConcatenator


@pytest.mark.skipif(
    not run_test_for_class(ColumnConcatenator),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_TimeSeriesConcatenator():
    """Test the time series concatenator."""
    X, y = load_basic_motions(split="train", return_X_y=True)

    # check that loaded dataframe is multivariate
    assert X.shape[1] > 1

    trans = ColumnConcatenator()

    Xt = trans.fit_transform(X)

    # check if transformed dataframe is univariate
    assert Xt.shape[1] == 1

    # check if number of time series observations are correct
    n_obs = np.sum([X.loc[0, col].shape[0] for col in X])
    assert Xt.iloc[0, 0].shape[0] == n_obs

    # check specific observations
    assert X.iloc[0, -1].iloc[-3] == Xt.iloc[0, 0].iloc[-3]
    assert X.iloc[0, 0].iloc[3] == Xt.iloc[0, 0].iloc[3]


@pytest.mark.skipif(
    not run_test_for_class(ColumnConcatenator),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("columns", [["z", "a"], ["c2", "c10"], [1, 0]])
def test_column_order_preserved(columns):
    """Test that columns are concatenated from left to right, not sorted by name."""
    idx = pd.MultiIndex.from_product([[0, 1], range(3)], names=["inst", "time"])
    X = pd.DataFrame(
        {columns[0]: [1, 2, 3, 4, 5, 6], columns[1]: [10, 20, 30, 40, 50, 60]},
        index=idx,
    )

    Xt = ColumnConcatenator().fit_transform(X)

    assert Xt.loc[0].iloc[:, 0].tolist() == [1, 2, 3, 10, 20, 30]
    assert Xt.loc[1].iloc[:, 0].tolist() == [4, 5, 6, 40, 50, 60]
