# -*- coding: utf-8 -*-
import numpy as np
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator


def test_TimeSeriesConcatenator():
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
