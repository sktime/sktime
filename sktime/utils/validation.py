import numpy as np
import pandas as pd


def check_y(instances, class_labels):
    if class_labels is not None:
        if len(class_labels) != instances.shape[0]:
            raise ValueError("instances not same length as class_labels")


def check_X_y(instances, class_labels):
    check_X(instances)
    check_y(instances, class_labels)


def check_X(instances):
    if not isinstance(instances, pd.DataFrame):
        raise ValueError("instances not in panda dataframe")

def check_ts_X_y(X, y):
    """
    Placeholder function for input validation.
    """
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_X_y(X, y, dtype=None, ensure_2d=False)
    return X, y


def check_ts_array(X):
    """
    Placeholder function for input validation.
    """
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_array(X, dtype=None, ensure_2d=False)
    return X


def check_equal_index(X):
    """
    Check if all time-series for a given column in a
    nested pandas DataFrame have the same index.

    Parameters
    ----------
    X : nested pandas DataFrame
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
            raise ValueError(f'Time series must contain at least 2 observations, but found: '
                             f'{len(first_index)} observations in column: {col}')

        # Check index for all rows.
        for i in range(1, X.shape[0]):
            index = X.iloc[i, c].index if hasattr(X.iloc[i, c], 'index') else np.arange(X.iloc[c, 0].shape[0])
            if not np.array_equal(first_index, index):
                raise ValueError(f'Found time series with unequal index in column {col}. '
                                 f'Input time-series must have the same index.')
        indexes.append(first_index)

    return indexes


def validate_fh(fh):
    """
    Validate forecasting horizon.

    Parameters
    ----------
    fh : list of int
        Forecasting horizon with steps ahead to predict.

    Returns
    -------
    fh : numpy array of int
        Sorted and validated forecasting horizon.
    """

    # Set default as one-step ahead
    if fh is None:
        return np.ones(1, dtype=np.int)

    # Check single integer
    elif np.issubdtype(type(fh), np.integer):
        return np.array([fh], dtype=np.int)

    # Check array-like input
    else:
        if isinstance(fh, list):
            if len(fh) < 1:
                raise ValueError(f"`fh` must specify at least one step, but found: "
                                 f"{type(fh)} of length {len(fh)}")
            if not np.all([np.issubdtype(type(h), np.integer) for h in fh]):
                raise ValueError('If `fh` is passed as a list, '
                                 'it has to be a list of integers')

        elif isinstance(fh, np.ndarray):
            if fh.ndim > 1:
                raise ValueError(f"`fh` must be a 1d array, but found: "
                                 f"{fh.ndim} dimensions")
            if len(fh) < 1:
                raise ValueError(f"`fh` must specify at least one step, but found: "
                                 f"{type(fh)} of length {len(fh)}")
            if not np.issubdtype(fh.dtype, np.integer):
                raise ValueError(
                    f'If `fh` is passed as an array, it has to be an array of '
                    f'integers, but found an array of dtype: {fh.dtype}')

        else:
            raise ValueError(f"`fh` has to be either a list or array of integers, or a single "
                             f"integer, but found: {type(fh)}")

        return np.asarray(np.sort(fh), dtype=np.int)
