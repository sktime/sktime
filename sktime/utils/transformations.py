import pandas as pd
import numpy as np


def tabularize(X, return_array=False):
    """
    Helper function to turn nested pandas dataframes into tabular data.
    :param X: nested pandas dataframe
    :param return_array : bool
        If True, returns a numpy array of the tabular data.
        If False, returns a pandas dataframe with row and column names.
    :return: tabular pandas dataframe
    """

    Xt = np.hstack([col.tolist() for _, col in X.items()])
    if return_array:
        return Xt

    Xt = pd.DataFrame(Xt)
    Xt.index = X.index
    columns = []
    for colname, col in X.items():
        tsindex = col.iloc[0].index if hasattr(col.iloc[0], 'index') else np.arange(col.iloc[0].shape[0])
        columns.extend([f'{colname}_{i}' for i in tsindex])
    Xt.columns = columns
    return Xt


tabularise = tabularize

