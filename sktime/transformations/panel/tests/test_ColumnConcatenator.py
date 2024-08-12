"""Column concatenator test code."""

import numpy as np
import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.compose import ColumnConcatenator


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
