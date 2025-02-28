"""Functions for reading time series data from .tsf files."""

__author__ = ["rakshitha123"]

__all__ = ["load_tsf_to_dataframe"]

from datetime import datetime

import pandas as pd

from sktime.datasets._readers_writers.utils import get_path
from sktime.datatypes import MTYPE_LIST_HIERARCHICAL, convert
from sktime.utils.strtobool import strtobool


def _convert_tsf_to_hierarchical(
    data: pd.DataFrame,
    metadata: dict,
    freq: str = None,
    value_column_name: str = "series_value",
) -> pd.DataFrame:
    """Convert the data from default_tsf to pd_multiindex_hier.

    Parameters
    ----------
    data : pd.DataFrame
        nested values dataframe
    metadata : Dict
        tsf file metadata
    freq : str, optional
        pandas compatible time frequency, by default None
        if not specified it's automatically mapped from the tsf frequency to a pandas
        frequency
    value_column_name: str, optional
        The name of the column that contains the values, by default "series_value"

    Returns
    -------
    pd.DataFrame
        sktime pd_multiindex_hier mtype
    """
    df = data.copy()

    if freq is None:
        freq_map = {
            "4_seconds": "4S",
            "minutely": "min",
            "10_minutes": "10min",
            "half_hourly": "30min",
            "hourly": "H",
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
            "quarterly": "QS",
            "yearly": "YS",
            None: None,
        }
        freq = freq_map[metadata["frequency"]]

    # create the time index
    if "start_timestamp" in df.columns:
        try:
            df["timestamp"] = df.apply(
                lambda x: pd.date_range(
                    start=x["start_timestamp"],
                    periods=len(x[value_column_name]),
                    freq=freq,
                ),
                axis=1,
            )
            drop_columns = ["start_timestamp"]
            has_time_index = True
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            # Parts of the yearly time series from M4 are to long to be encoded
            # with unit="ns" in pandas date_range. Other units cause problems when
            # creating the final index.
            # Thus, we ignore the datetime index if the time series is too long.
            drop_columns, has_time_index = create_range_index(value_column_name, df)
    else:
        drop_columns, has_time_index = create_range_index(value_column_name, df)

    # pandas implementation of multiple column explode
    # can be removed and replaced by explode if we move to pandas version 1.3.0
    columns = [value_column_name, "timestamp"]
    index_columns = [c for c in list(df.columns) if c not in drop_columns + columns]
    result = pd.DataFrame({c: df[c].explode() for c in columns})
    df = df.drop(columns=columns + drop_columns).join(result)
    if df["timestamp"].dtype == "object":
        df = df.astype({"timestamp": "int64"})
    df = df.set_index(index_columns + ["timestamp"])
    df = df.astype({value_column_name: "float"}, errors="ignore")

    if has_time_index:
        try:
            df.index.levels[-1].freq = freq
        except ValueError:
            # If the datimeindex is not consecutive setting frequency will fail..
            pass
    return df


def create_range_index(value_column_name, df):
    df["timestamp"] = df.apply(
        lambda x: pd.RangeIndex(start=0, stop=len(x[value_column_name])), axis=1
    )
    drop_columns = []
    has_time_index = False
    return drop_columns, has_time_index


# TODO: depreciate this and rename it load_from_tsf_to_dataframe for consistency
def load_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
    return_type="pd_multiindex_hier",
):
    """Convert the contents in a .tsf file into a dataframe.

    This code was extracted from
    https://github.com/rakshitha123/TSForecasting/blob
    /master/utils/data_loader.py.

    Parameters
    ----------
    full_file_path_and_name: str
        The full path to the .tsf file.
    replace_missing_vals_with: str, default="NAN"
        A term to indicate the missing values in series in the returning dataframe.
    value_column_name: str, default="series_value"
        Any name that is preferred to have as the name of the column containing series
        values in the returning dataframe.
    return_type : str - "pd_multiindex_hier" (default), "tsf_default", or valid sktime
        mtype string for in-memory data container format specification of the
        return type:
        - "pd_multiindex_hier" = pd.DataFrame of sktime type ``pd_multiindex_hier``
        - "tsf_default" = container that faithfully mirrors tsf format from the original
            implementation in: https://github.com/rakshitha123/TSForecasting/
            blob/master/utils/data_loader.py.
        - other valid mtype strings are Panel or Hierarchical mtypes in
            datatypes.MTYPE_REGISTER. If Panel or Hierarchical mtype str is given, a
            conversion to that mtype will be attempted
        For tutorials and detailed specifications, see
        examples/AA_datatypes_and_datasets.ipynb

    Returns
    -------
    loaded_data: pd.DataFrame
        The converted dataframe containing the time series.
    metadata: dict
        The metadata for the forecasting problem. The dictionary keys are:
        "frequency", "forecast_horizon", "contain_missing_values",
        "contain_equal_length"
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    full_file_path_and_name = get_path(full_file_path_and_name, ".tsf")

    with open(full_file_path_and_name, encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. "
                                "Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. "
                            "Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set "
                                "of comma separated numeric values."
                                "At least one numeric value should be there "
                                "in a series. "
                                "Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. "
                                "A given series should contains a set "
                                "of comma separated numeric values."
                                "At least one numeric value should be there "
                                "in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                # Currently, the code supports only
                                # numeric, string and date types.
                                # Extend this as required.
                                raise Exception("Invalid attribute type.")

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        # metadata dict
        metadata = dict(
            zip(
                (
                    "frequency",
                    "forecast_horizon",
                    "contain_missing_values",
                    "contain_equal_length",
                ),
                (
                    frequency,
                    forecast_horizon,
                    contain_missing_values,
                    contain_equal_length,
                ),
            )
        )

        if return_type != "default_tsf":
            loaded_data = _convert_tsf_to_hierarchical(
                loaded_data, metadata, value_column_name=value_column_name
            )
            if (
                loaded_data.index.nlevels == 2
                and return_type not in MTYPE_LIST_HIERARCHICAL
            ):
                loaded_data = convert(
                    loaded_data, from_type="pd-multiindex", to_type=return_type
                )
            else:
                loaded_data = convert(
                    loaded_data, from_type="pd_multiindex_hier", to_type=return_type
                )

        return loaded_data, metadata
