from sktime.utils.transformations import tabularize
from sktime.utils.testing import generate_df_from_array
import numpy as np
import pandas as pd


def test_tabularize():
    n_obs_X = 20
    n_cols_X = 3
    X = generate_df_from_array(np.random.normal(size=n_obs_X), n_rows=10, n_cols=n_cols_X)

    # Test single series input.
    Xt = tabularize(X.iloc[:,0], return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == n_obs_X

    Xt = tabularize(X.iloc[:, 0])
    assert Xt.index.equals(X.index)

    # Test dataframe input with columns having series of different length.
    n_obs_Y = 13
    n_cols_Y = 2
    Y = generate_df_from_array(np.random.normal(size=n_obs_Y), n_rows=10, n_cols=n_cols_Y)
    X = pd.concat([X, Y], axis=1)

    Xt = tabularize(X, return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == (n_cols_X * n_obs_X) + (n_cols_Y * n_obs_Y)

    Xt = tabularize(X)
    assert Xt.index.equals(X.index)
