import pandas as pd


def convert_pandas_to_listDataset(pd_dataframe: pd.DataFrame, is_single: bool = False):
    """Convert a given pandas DataFrame to a gluonTS ListDataset.

    Parameters
    ----------
    pd_dataframe : pd.DataFrame
        A valid pandas DataFrame

    is_single : bool (default=False)
        True if there is only 1 time series, false if there are multiple series
        (Allows reusability with _series and _panel implementations)

    Returns
    -------
    gluonts.dataset.common.ListDataset
        A gluonTS ListDataset formed from `pd_dataframe`

    Raises
    ------
    ValueError
        If is_single is True, but multiple rows of entries exist in `pd_dataframe`

    Example
    --------
    >>> import pandas as pd
    >>> cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(2)]
    >>> Xlist = [
    ...    pd.DataFrame([[0, 0, 1, 4], [0, 1, 2, 5], [0, 2, 3, 6]], columns=cols),
    ...    pd.DataFrame([[1, 0, 1, 4], [1, 1, 2, 55], [1, 2, 3, 6]], columns=cols),
    ...    pd.DataFrame([[2, 0, 1, 42], [2, 1, 2, 5], [2, 2, 3, 6]], columns=cols),
    ... ]

    >>> X = pd.concat(Xlist)

    # Setting the timepoints in a Pandas acceptable format
    >>> X['timepoints'] = pd.date_range(start='2023-01-01', periods=len(X), freq='D')

    # Resetting the indexes to create the multiindexed DF
    >>> X = X.set_index(["instances", "timepoints"])

    # Finally, converting to a GluonTS ListDataset!
    >>> from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset
    >>> X_list_dataset = convert_pandas_to_listDataset(X)
    """
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName

    # Assert the validity of the is_single parameter
    if is_single and pd_dataframe.shape[0] > 1:
        raise ValueError("`is_single` is True but the DataFrame has multiple rows!")

    # This list will store all individual time series
    dataset = []

    # By maintaining 2 levels in the DataFrame's indices
    # we can access each series and its timestep values with ease!
    for _, data in pd_dataframe.groupby(level=0):
        # Getting the starting time for each series (assumed to be min value)
        start_datetime = data.index.get_level_values(level=1).min()

        # Isolating multivariate values for each time series
        target_column = data.columns[:]
        target_values = data[target_column]

        dataset.append(
            {
                FieldName.START: start_datetime,
                FieldName.TARGET: target_values,
            }
        )

    # Obtain the total amount of timesteps to assist with inferring frequency
    n_steps = pd_dataframe.index.get_level_values(level=1).nunique()
    fr = pd.infer_freq(pd_dataframe.index.get_level_values(level=1)[:n_steps])

    # Converting the dataset to a GluonTS ListDataset
    list_dataset = ListDataset(
        dataset,
        freq=fr,
        one_dim_target=False,
    )

    return list_dataset


def convert_listDataset_to_pandas(list_dataset):
    """Convert a given gluonTS ListDataset to a pandas DataFrame.

    Parameters
    ----------
    list_dataset : ListDataset
        A valid GluonTS ListDataset to be converted to a pd.DataFrame

    Returns
    -------
    pd.DataFrame
        A valid pandas DataFrame
    """
    dfs = []

    # Processing each given time series
    for id, data in enumerate(list_dataset):
        # Extracting important features
        target_values = data["target"]
        start_datetime = data["start"]

        freq = start_datetime.freq

        # Creating new indices based on the start time and the frequency
        index = pd.date_range(
            start=start_datetime.to_timestamp(), freq=freq, periods=len(target_values)
        )

        df = pd.DataFrame(target_values, index=index)

        # Adding in the series_id index to the DataFrame and adding it to `dfs`
        df.insert(0, "series", [f"series_{id}"] * len(df))
        dfs.append(df)

    # Concatenating all the given dataframes
    dfs = pd.concat(dfs, ignore_index=False)

    # Setting the index hierarchy properly
    dfs = dfs.set_index(["series", dfs.index]).rename_axis(["series", "timestamp"])

    # Adding column values
    columns = [f"value_{i}" for i in range(1, dfs.shape[1] + 1, 1)]
    dfs.columns = columns

    return dfs
