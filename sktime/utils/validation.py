'''
validation utilities for sktime
'''
# build on top of sklearn
import numpy as np

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
    Check for each column in dataframe if all series have the same index.

    :param X:
    :return:
    """

    # TODO assumes columns are typed (i.e. all rows for a given column have the same type)
    # TODO only handles series columns, raises error for columns with primitives

    indexes = []
    # Check index for each column separately.
    for c, col in enumerate(X.columns):

        # Get index from first row, can be either pd.Series or np.array.
        first_index = X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index') else np.arange(X.iloc[c, 0].shape[0])

        # Series must contain at least 2 observations, otherwise should be primitive.
        if len(first_index) < 2:
            raise ValueError(f'Time series must contain at least 2 observations, '
                             f'found time series in column {col} with less than 2 observations')

        # Check index for all rows.
        for i in range(1, X.shape[0]):
            index = X.iloc[i, c].index if hasattr(X.iloc[i, c], 'index') else np.arange(X.iloc[c, 0].shape[0])
            if not np.array_equal(first_index, index):
                raise ValueError(f'Found time series with unequal index in column {col}. '
                                 f'Input time-series must have the same index.')
        indexes.append(first_index)

    return indexes
