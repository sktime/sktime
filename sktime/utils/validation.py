'''
validation utilities for sktime
'''
# build on top of sklearn
from sklearn.utils.validation import check_X_y


def check_ts_X_y(X, y):
    '''
    use preexisting ones with bypass (temporarily)
    '''
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_X_y(X, y, dtype=None, ensure_2d=False)
    return X, y


def check_ts_array(X):
    '''
    use preexisting ones with bypass (temporarily)
    '''
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_array(X, dtype=None, ensure_2d=False)
    return X


def check_equal_index(X):
    """
    Check if all series in the same column have the same index.
    """
    n_rows, n_cols = X.shape
    indexes = []
    for c in range(n_cols):
        first_index = X.iloc[0, c].index
        for i in range(1, n_rows):
            if not first_index.equals(X.iloc[i, c].index):
                raise ValueError(f'Found time series with unequal index in column {c}. '
                                 'Input time-series must have the same index.')
        indexes.append(first_index)
    return indexes
