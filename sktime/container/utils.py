import numpy as np
import pandas as pd
from sktime.container import TimeArray
from sktime.container.array import from_list, from_pandas

def convert_to_timearray(obj, time_index=None):
    """
    Turns a numpy array into TimeArray if it is 2-dimensional, or leaves it unchanged if has fewer dimensions. Throws
    an error for higher dimensions.

    Parameters
    ----------
    obj : np.ndarray
        numpy array to be converted into a TimeArray

    Returns
    -------
    TimeArray or 1-D np.ndarray
        if 2-D, then the converted input, otherwise the input is passed through without changing it
    """

    if isinstance(obj, pd.DataFrame):
        obj = from_pandas(obj)

    elif isinstance(obj, pd.Series):
        # Only convert series if they are nested, i.e. a series of series
        row_is_series = [True if isinstance(x, pd.Series) else False for x in obj]
        if np.all(row_is_series):
            obj = from_pandas(obj)

    elif isinstance(obj, list):
        obj = from_list(obj)

    elif isinstance(obj, np.ndarray) and obj.ndim == 2:
            obj = TimeArray(obj, time_index)

    return obj