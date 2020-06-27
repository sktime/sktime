import numpy as np
import pandas as pd
import pytest
import math

from sktime.exceptions import NotFittedError
from sktime.transformers.series_as_features.derivative \
    import DerivativeTransformer
from sktime.utils.data_container import tabularize
from sktime.utils._testing import generate_df_from_array

            
# Check that NotFittedError is thrown if someone attempts to
# transform before calling fit
def test_early_trans_fail():

    X = generate_df_from_array(np.ones(10), n_rows=1, n_cols=1)
    d = DerivativeTransformer()

    with pytest.raises(NotFittedError):
        d.transform(X)

# Check the transformer has changed the data correctly.   
def test_output_of_transformer():

    X = generate_df_from_array(np.array([4,6,10,12,8,6,5,5]), n_rows=1, n_cols=1)

    d = DerivativeTransformer().fit(X)
    res = d.transform(X)
    orig = convert_list_to_dataframe([[-2,-4,-2,4,2,1,0]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res,orig)
    
    X = generate_df_from_array(np.array([-5,2.5,1,3,10,-1.5,6,12,-3]), n_rows=1, n_cols=1)
    d = d.fit(X)
    res = d.transform(X)
    orig = convert_list_to_dataframe([[-7.5,1.5,-2,-7,11.5,-7.5,-6,15]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res,orig)
    
@pytest.mark.parametrize("orig_series_length,corr_series_length", [(12,11),(5,4),(56,55)])
def test_output_dimensions(orig_series_length, corr_series_length):

    X = generate_df_from_array(np.ones(orig_series_length), n_rows=10, n_cols=1)
    
    d = DerivativeTransformer().fit(X)
    res = d.transform(X)
    
    # get the dimension of the generated dataframe.
    act_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]
    
    assert act_time_series_length == corr_series_length
    assert num_rows == 10
    assert num_cols == 1

# This is to check that Derivative produces the same result along each dimension
def test_derivative_performs_correcly_along_each_dim():

    X = generate_df_from_array(np.array([1,2,3,4,5,6,7,8,9,10]), n_rows = 1, n_cols=2)
    
    d = DerivativeTransformer().fit(X)
    res = d.transform(X)
    orig = convert_list_to_dataframe([[-1,-1,-1,-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1,-1,-1]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res,orig)

def convert_list_to_dataframe(list_to_convert):
    # Convert this into a panda's data frame
    df = pd.DataFrame()
    for i in range(len(list_to_convert)):
        inst = list_to_convert[i]
        data = []
        data.append(pd.Series(inst))
        df[i] = data
        
    return df
    
"""
for some reason, this is how you check that two dataframes are equal.
"""
def check_if_dataframes_are_equal(df1,df2):
    from pandas.testing import assert_frame_equal
    
    try:
        assert_frame_equal(df1, df2)
        return True
    except AssertionError as e: 
        return False