import numpy as np


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