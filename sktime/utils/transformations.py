import pandas as pd
import numpy as np
from .validation import validate_fh

__author__ = "Markus Löning"


def tabularize(X, return_array=False):
    """Convert nested pandas DataFrames or Series with numpy arrays or pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires series to be have the same index.

    Parameters
    ----------
    X : nested pandas DataFrame or nested Series
    return_array : bool, optional (default=False)
        - If True, returns a numpy array of the tabular data.
        - If False, returns a pandas dataframe with row and column names.

    Returns
    -------
     Xt : pandas DataFrame
        Transformed dataframe in tabular format
    """

    # TODO does not handle dataframes with nested series columns *and* standard columns containing only primitives

    if X.ndim == 1:
        Xt = np.array(X.tolist())
    else:
        Xt = np.hstack([col.tolist() for _, col in X.items()])

    if return_array:
        return Xt

    Xt = pd.DataFrame(Xt)
    Xt.index = X.index
    if X.ndim == 1:
        tsindex = X.iloc[0].index if hasattr(X.iloc[0], 'index') else np.arange(X.iloc[0].shape[0])
        columns = [f'{X.name}_{i}' for i in tsindex]
    else:
        columns = []
        for colname, col in X.items():
            tsindex = col.iloc[0].index if hasattr(col.iloc[0], 'index') else np.arange(col.iloc[0].shape[0])
            columns.extend([f'{colname}_{i}' for i in tsindex])
    Xt.columns = columns
    return Xt


def detabularize(X, return_arrays=False):
    """Convert tabular pandas DataFrame with only primitives in cells into nested pandas DataFrame with a single column.

    Parameters
    ----------
    X : nested pandas DataFrame or nested Series
    return_arrays : bool, optional (default=False)
        - If True, returns a numpy arrays within cells of nested pandas DataFrame.
        - If False, returns a pandas Series within cells.

    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe in nested format
    """
    n_rows = X.shape[0]
    container = np.asarray if return_arrays else pd.Series
    Xt = [container(X.iloc[i, :].values) for i in range(n_rows)]
    return pd.DataFrame(pd.Series(Xt))


tabularise = tabularize


detabularise = detabularize


def select_times(X, times):
    """Select times from time series within cells of nested pandas DataFrame.

    Parameters
    ----------
    X : nested pandas DataFrame or nested Series
    times : numpy ndarray of times to select from time series

    Returns
    -------
    Xt : pandas DataFrame
        pandas DataFrame in nested format containing only selected times
    """
    Xt = detabularise(tabularise(X).iloc[:, times])
    Xt.columns = X.columns
    return Xt


def concat_nested_arrays(arrs, return_arrays=False):
    """
    Helper function to nest tabular arrays from nested list of arrays.

    Parameters
    ----------
    arrs : list of numpy arrays
        Arrays must have the same number of rows, but can have varying number of columns.
    return_arrays: bool, optional (default=False)
        - If True, return pandas DataFrame with nested numpy arrays.
        - If False, return pandas DataFrame with nested pandas Series.

    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe with nested column for each input array.
    """
    if return_arrays:
        Xt = pd.DataFrame(np.column_stack(
            [pd.Series([np.array(vals) for vals in interval])
             for interval in arrs]))
    else:
        Xt = pd.DataFrame(np.column_stack(
            [pd.Series([pd.Series(vals) for vals in interval])
             for interval in arrs]))
    return Xt


class RollingWindowSplit:
    """Rolling window iterator that allows to split time series index into two windows,
    one containing observations used as feature data and one containing observations used as
    target data to be predicted. The target window has the length of the given forecasting horizon.

    Parameters
    ----------
    window_length : int
        Length of rolling window
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.
    """

    def __init__(self, window_length=None, fh=None):
        # TODO input checks
        if window_length is not None:
            if not np.issubdtype(type(window_length), np.integer):
                raise ValueError(f"Window length must be an integer, but found: {type(window_length)}")

        self.window_length = window_length
        self.fh = validate_fh(fh)

        # Attributes updated in split
        self.n_splits_ = None
        self.window_length_ = None

    def split(self, data):
        """
        Split data using rolling window.

        Parameters
        ----------
        data : ndarray
            1-dimensional array of time series index to split.

        Yields
        ------
        features : ndarray
            The indices of the feature window
        targets : ndarray
            The indices of the target window
        """

        # Input checks.
        if not isinstance(data, np.ndarray) and (data.ndim == 1):
            raise ValueError(f"Passed data has to be 1-d numpy array, but found data of type: {type(data)} with "
                             f"{data.ndim} dimensions")

        n_obs = data.shape[0]
        max_fh = self.fh[-1]  # furthest step ahead, assume fh is sorted

        # Set default window length to sqrt of series length
        self.window_length_ = int(np.sqrt(n_obs)) if self.window_length is None else self.window_length

        if (self.window_length_ + max_fh) > n_obs:
            raise ValueError("Window length and forecast horizon cannot be longer than data")

        # Iterate over windows
        start = self.window_length_
        stop = n_obs - max_fh + 1
        self.n_splits_ = stop - start

        for window in range(start, stop):
            features = data[window - self.window_length_:window]
            targets = data[window + self.fh - 1]
            yield features, targets

    def get_n_splits(self):
        """
        Return number of splits.
        """
        return self.n_splits_

    def get_window_length(self):
        """
        Return the window length.
        """
        return self.window_length_
