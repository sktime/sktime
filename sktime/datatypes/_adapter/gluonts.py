import pandas as pd


def convert_pandas_to_listDataset(pd_dataframe: pd.DataFrame):
    """Convert a given pandas DataFrame to a gluonTS ListDataset.

    Parameters
    ----------
    pd_dataframe : pd.DataFrame
        A valid pandas DataFrame

    Returns
    -------
    list_dataset: list
        A list of dict where each dict represents a time series with the following keys:

        - "start": The starting timestamp.
        - "target": The values of the time series, as a numpy array.

    Examples
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
    # use numpy to implement abstraction of gluonTS ListDataset
    import numpy as np

    # For non-multiindexed DataFrames
    if not isinstance(pd_dataframe.index, pd.MultiIndex):
        start_datetime = pd_dataframe.index[0]

        target_columns = pd_dataframe.columns.difference(["instances", "timepoints"])
        target_values = pd_dataframe[target_columns].to_numpy().astype(np.float32)

        if isinstance(pd_dataframe.index, pd.DatetimeIndex):
            freq = pd_dataframe.index.inferred_freq

        elif isinstance(pd_dataframe.index, pd.PeriodIndex):
            freq = pd_dataframe.index.freq

        else:
            start_datetime = pd.Timestamp(start_datetime)
            freq = "D"

        return [
            {
                "start": pd.Period(start_datetime, freq),
                "target": target_values,
            }
        ]

    list_dataset = []

    # Obtain the total amount of timesteps to assist with inferring frequency
    time_index = pd_dataframe.index.get_level_values(1)
    freq = None

    if isinstance(time_index, pd.DatetimeIndex):
        freq = time_index.inferred_freq

    elif isinstance(time_index, pd.PeriodIndex):
        freq = time_index.freq

    if freq is None:
        freq = "D"

    # By maintaining 2 levels in the DataFrame's indices
    # we can access each series and its timestep values with ease!
    for _, data in pd_dataframe.groupby(level=0):
        data = data.reset_index(level=0, drop=True)

        # Getting the starting time for each series
        start_datetime = data.index[0]

        if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            start_datetime = pd.Timestamp(start_datetime)

        # Isolating multivariate values for each time series
        target_column = data.columns.difference(["instances"])

        target_values = data[target_column].to_numpy().astype(np.float32)

        list_dataset.append(
            {
                "start": pd.Period(start_datetime, freq),
                "target": target_values,
            }
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


def convert_pandas_dataframe_to_pandasDataset(pd_dataframe: pd.DataFrame):
    """Convert a given pandas DataFrame to a gluonTS PandasDataset.

    Parameters
    ----------
    pd_dataframe : pd.DataFrame
        A valid pandas DataFrame with the index as pd.DatetimeIndex

    Returns
    -------
    gluonts.dataset.common.PandasDataset
        A convert gluonTS PandasDataset

    Raises
    ------
    ValueError
        Raises a valueError if the index is not of instance `DatetimeIndex`.
    """
    # Importing required libraries
    from gluonts.dataset.pandas import PandasDataset

    # Checking for index validity
    if not isinstance(pd_dataframe.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("The dataframe must have a DateTimeIndex or PeriodIndex!")

    return PandasDataset(pd_dataframe)


def convert_pandasDataset_to_pandas_dataframe(pandasDataset):
    """Convert a PandasDataset into a valid Pandas DataFrame.

    Parameters
    ----------
    pandasDataset : GluonTS PandasDataset
        A valid gluonTS pandasDataset

    Raises
    ------
    ValueError
        Raises a value error if the pandasDataset is not a valid
        GluonTS PandasDataset

    Returns
    -------
    pd.DataFrame
        Returns the pandas DataFrame stored in the PandasDataset
    """
    from gluonts.dataset.pandas import PandasDataset

    # Checking for a valid data type
    if not isinstance(pandasDataset, PandasDataset):
        raise ValueError("The passed object must be a valid GluonTS PandasDataset!")

    return pandasDataset._data_entries.iterable[0][1]


def convert_pandas_multiindex_to_pandasDataset(
    pd_dataframe: pd.DataFrame,
    target=None,
    item_id="instances",
    timepoints="timepoints",
    freq="D",
):
    """Convert a given pandas DataFrame (multiindex) to a gluonTS PandasDataset.

    Parameters
    ----------
    pd_dataframe : pd.DataFrame
        A pd.DataFrame with each column corresponding to a time series

    target : str | list[str]
        The column that corresponds to target values.
        If no value provided, all column(s) assumed to be targets.

    item_id : str (optional, default="instances")
        A column dedicated to time series labels.
        "Instances" by default

    timepoints: str (optional, default="timepoints")
        The level name corresponding to the timepoints.
        "timepoints" by default

    freq : str (optional, default="D")
        The frequency of the given timepoints.
        "D" by default

    Returns
    -------
    gluonts.dataset.common.PandasDataset
        A gluonTS PandasDataset formed from `pd_dataframe`

    Raises
    ------
    ValueError
        If the given DataFrame does not have valid Pandas timestamps
    """
    from gluonts.dataset.pandas import PandasDataset

    # Assume all columns to be valid targets here
    if target is None:
        target = list(pd_dataframe.columns)

    # Converting to float32 for MPS compatibility
    float64_cols = list(pd_dataframe.select_dtypes(include="float64"))
    pd_dataframe[float64_cols] = pd_dataframe[float64_cols].astype("float32")

    # Assists with finding level-based values
    pd_dataframe = pd_dataframe.reset_index()

    if not isinstance(
        pd_dataframe[timepoints][0], pd._libs.tslibs.timestamps.Timestamp
    ):
        pd_dataframe[timepoints] = pd_dataframe[timepoints].apply(
            lambda x: pd.Timestamp(x)
        )

    if item_id not in pd_dataframe:
        pd_dataframe = pd_dataframe.rename({pd_dataframe.columns[0]: item_id}, axis=1)

    # Finally, we create and return a PandasDataset
    return PandasDataset.from_long_dataframe(
        pd_dataframe, item_id=item_id, timestamp=timepoints, target=target, freq=freq
    )


def convert_pandasDataset_to_pandas(
    pandasDataset, item_id="instances", timepoints="timepoints", convert_time=True
):
    """Convert a GluonTS PandasDataset to a pd.DataFrame.

    Parameters
    ----------
    pandasDataset : gluonts.dataset.pandas.PandasDataset
        A gluonTS PandasDataset

    item_id : str (optional, default="instances")
        A column dedicated to time series labels, "instances" by default

    timepoints: str (optional, default="timepoints")
        The name of the timepoints column, "timepoints" by default

    convert_time : bool (default=False)
        If True, converts the timepoints' ns back into integers

    Returns
    -------
    Returns a valid pandas DataFrame
    """
    # Extracting the DataFrame
    iterable = pandasDataset._data_entries.iterable.iterable

    # Merging the DataFrame
    dfs = [df[1] for df in iterable]
    dfs = pd.concat(dfs, keys=range(len(dfs)))

    # Remove duplicate timepoints if it exists
    dfs = dfs.drop("timepoints", axis=1)
    merged_df = dfs.reset_index()

    # If True, converts timestamps' nanoseconds back into integers
    if convert_time:
        merged_df["timepoints"] = merged_df["timepoints"].apply(lambda x: x.nanosecond)

    merged_df = merged_df.set_index([item_id, timepoints])
    merged_df = merged_df.drop("level_0", axis=1)

    # Converting to float32 for MPS compatibility
    float64_cols = list(merged_df.select_dtypes(include="float64"))
    merged_df[float64_cols] = merged_df[float64_cols].astype("float32")

    return merged_df


def convert_pandas_collection_to_pandasDataset(
    collection_dataframe, timepoints="timepoints", freq="D", target="target"
):
    """Convert a list of pd.DataFrames or dict of pd.DataFrames to a PandasDataset.

    Parameters
    ----------
    collection_dataframe : list[pd.DataFrame] | dict[pd.DataFrame]
        A list or dictionary of pandas DataFrames
        If dictionary, key = item_id, value = pd.DataFrame

    timepoints : str
        The column corresponding to the timepoints (default='timepoints')

    freq : str
        The frequency of the timepoints (default='D')

    target : str
        The column corresponding to the target values column (default="target")

    Returns
    -------
    gluonts.dataset.common.PandasDataset
        A gluonTS PandasDataset formed from `collection_dataframe`

    Raises
    ------
    ValueError
        If `collection_dataframe` is not a list or a dict,
        if any entity is not of the format `pd.DataFrame`
    """
    # Importing required libraries
    from gluonts.dataset.pandas import PandasDataset

    # Update target to the default value if it is None
    if target is None:
        target = "target"

    # Checking for the dictionary-format
    if type(collection_dataframe) is dict:
        for key, df in collection_dataframe.items():
            # Check for pd.DataFrame format
            if type(df) is not pd.DataFrame:
                raise ValueError(f"The value of {key} is not a pandas DataFrame!")

            # Check for a target value
            if isinstance(target, list):
                if not set(target).issubset(df.columns):
                    raise ValueError(
                        f"The DataFrame at {key} does not have a target column!"
                    )

            elif target not in df.columns:
                raise ValueError(
                    f"The DataFrame at {key} does not have a target column!"
                )

            # Converting to float32 for MPS compatibility
            float64_cols = list(df.select_dtypes(include="float64"))
            df[float64_cols] = df[float64_cols].astype("float32")

        return PandasDataset(
            dataframes=collection_dataframe,
            timestamp=timepoints,
            freq=freq,
            target=target,
        )

    # Checking for the list-format
    elif type(collection_dataframe) is list:
        for idx, df in enumerate(collection_dataframe):
            # Check for valid timestamps
            if timepoints not in df.columns and not isinstance(
                df.index, pd.DatetimeIndex
            ):
                raise ValueError(
                    f"The DataFrame at the {idx} index does not have the "
                    + f"required '{timepoints}' columns!"
                )

            # Converting to float32 for MPS compatibility
            float64_cols = list(df.select_dtypes(include="float64"))
            df[float64_cols] = df[float64_cols].astype("float32")

        return PandasDataset(
            dataframes=collection_dataframe,
            timestamp=timepoints,
            freq=freq,
            target=target,
        )

    else:
        raise ValueError("Expected format of dict[pd.DataFrame] or list[pd.DataFrame]")
