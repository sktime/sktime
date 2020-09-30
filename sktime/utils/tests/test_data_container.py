import numpy as np
import pandas as pd
from sktime.utils.data_container import nested_to_3d_numpy
from sktime.utils.data_container import from_3d_numpy_to_nested
from sktime.utils.data_container import is_nested_dataframe


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
