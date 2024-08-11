"""Util function for reading long format files."""

__author__ = ["TonyBagnall", "AidenRushbrooke", "Markus Löning"]
__all__ = ["load_from_long_to_dataframe"]

import pandas as pd

from sktime.datasets._readers_writers.utils import get_path
from sktime.datatypes._panel._convert import from_long_to_nested


# TODO: original author didn't add test for this function, for research purposes?
def load_from_long_to_dataframe(full_file_path_and_name, separator=","):
    """Load data from a long format file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .csv file to read.
    separator: str
        The character that the csv uses as a delimiter

    Returns
    -------
    DataFrame
        A dataframe with sktime-formatted data
    """
    full_file_path_and_name = get_path(full_file_path_and_name, ".csv")

    data = pd.read_csv(full_file_path_and_name, sep=separator, header=0)
    # ensure there are 4 columns in the long_format table
    if len(data.columns) != 4:
        raise ValueError("dataframe must contain 4 columns of data")
    # ensure that all columns contain the correct data types
    if (
        not data.iloc[:, 0].dtype == "int64"
        or not data.iloc[:, 1].dtype == "int64"
        or not data.iloc[:, 2].dtype == "int64"
        or not data.iloc[:, 3].dtype == "float64"
    ):
        raise ValueError("one or more data columns contains data of an incorrect type")

    data = from_long_to_nested(data)
    return data
