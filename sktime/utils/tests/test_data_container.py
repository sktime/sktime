import numpy as np
import pandas as pd
from sktime.utils._testing import generate_df_from_array
from sktime.utils.data_container import (
    is_nested_dataframe,
    tabularize,
    from_3d_numpy_to_nested,
    nested_to_3d_numpy
)

def test_tabularize():
    n_obs_X = 20
    n_cols_X = 3
    X = generate_df_from_array(np.random.normal(size=n_obs_X), n_rows=10,
                               n_cols=n_cols_X)

    # Test single series input.
    Xt = tabularize(X.iloc[:, 0], return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == n_obs_X

    Xt = tabularize(X.iloc[:, 0])
    assert Xt.index.equals(X.index)

    # Test dataframe input with columns having series of different length.
    n_obs_Y = 13
    n_cols_Y = 2
    Y = generate_df_from_array(np.random.normal(size=n_obs_Y), n_rows=10,
                               n_cols=n_cols_Y)
    X = pd.concat([X, Y], axis=1)

    Xt = tabularize(X, return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == (n_cols_X * n_obs_X) + (n_cols_Y * n_obs_Y)

    Xt = tabularize(X)
    assert Xt.index.equals(X.index)

def test_nested_to_3d_numpy():
    """Test the nested_to_3d_numpy() function.
    """

    df = pd.DataFrame({'var_1': [pd.Series([1, 2]), pd.Series([3, 4])],
                       'var_2': [pd.Series([5, 6]), pd.Series([7, 8])]})
    array = nested_to_3d_numpy(df)
    assert isinstance(array, np.ndarray)


def test_from_3d_numpy_to_nested():
    """Test the from_3d_numpy_to_nested() function.
    """

    array = np.random.normal(size=(5, 12, 2))
    nested = from_3d_numpy_to_nested(array)
    assert is_nested_dataframe(nested)
