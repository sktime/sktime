import numpy as np
import pandas as pd


def check_ts_X_y(X, y):
    """Placeholder function for input validation.
    """
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_X_y(X, y, dtype=None, ensure_2d=False)
    return X, y


def check_ts_array(X):
    """Placeholder function for input validation.
    """
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_array(X, dtype=None, ensure_2d=False)
    return X


def check_equal_index(X):
    """
    Function to check if all time-series for a given column in a nested pandas DataFrame have the same index.

    Parameters
    ----------
    param X : nested pandas DataFrame
        Input dataframe with time-series in cells.

    Returns
    -------
    indexes : list of indixes
        List of indixes with one index for each column
    """
    # TODO handle 1d series, not only 2d dataframes
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


def check_forecasting_horizon(fh):
    if isinstance(fh, list):
        if not np.all([np.issubdtype(type(h), np.integer) for h in fh]):
            raise ValueError('If the forecasting horizon ``fh`` is passed as a list, it has to be a list of integers')
    elif isinstance(fh, np.ndarray):
        if not np.issubdtype(fh.dtype, np.integer):
            raise ValueError(f'If the forecasting horizon ``fh`` is passed as an array, it has to be an array of '
                             f'integers, but found an array of dtype: {fh.dtype}')
    elif np.issubdtype(type(fh), np.integer):
        fh = [fh]
    else:
        raise ValueError(f"The forecasting horizon ``fh`` has to be either a list or array of integers, or a single "
                         f"integer, but found: {type(fh)}")
    return np.sort(fh)


def check_y_forecasting(y):
    # check if pandas series
    if not isinstance(y, pd.Series):
        raise ValueError(f'y must be pandas series, but found: {type(y)}')

    # check if single row
    n_rows = y.shape[0]
    if n_rows > 1:
        raise ValueError(f'y must consist of a single row, but found: {n_rows} rows')

    # check if contained time series is either pandas series or numpy array
    s = y.iloc[0]
    if not isinstance(s, (np.ndarray, pd.Series)):
        raise ValueError(f'y must contain a pandas series or numpy array, but found: {type(s)}.')
