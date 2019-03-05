import pandas as pd
import numpy as np


def tabularize(X):
    """
    Helper function to turn nested pandas dataframes into tabular data.
    :param X: nested pandas dataframe
    :return: tabular pandas dataframe
    """
    Xt = pd.DataFrame(np.hstack([col.tolist() for _, col in X.items()]))
    Xt.index = X.index
    columns = []
    for colname, col in X.items():
        tsindex = col.iloc[0].index if hasattr(col.iloc[0], 'index') else np.arange(col.iloc[0].shape[0])
        columns.extend([f'{colname}_{i}' for i in tsindex])
    Xt.columns = columns
    return Xt


tabularise = tabularize

