# -*- coding: utf-8 -*-
import pandas as pd
import re

__all__ = ["csv_to_ts_df"]

__author__ = ["Christopher Holder"]


def csv_to_ts_df(csv_df, group_by):
    """
    Loads and then puts csv data into a dataframe

    Parameters
    ----------
    path_to_csv: str
        The full pathname for the .csv file to read
    group_by_index: str
        The name of the column to group by

    Returns
    -------
    ts_df: dataframe
        Dataframe containing the converted csv file data to ts dataframe. This
        consists of the same column headers (excluding the group by column),
        and each row value being a pandas series
    csv_group_by_column_vals: arr
        Array containing a list of unique group by. The order is the same as
        the dataframe (i.e. ts_df index 0 group by label is
        csv_group_by_column_vals index 0)
    """
    csv_df_columns = csv_df.columns
    csv_group_by_column_vals = list(set(csv_df[group_by]))
    csv_df_columns = csv_df_columns.drop(group_by)
    ts_df = pd.DataFrame(columns=csv_df_columns)
    for val in csv_group_by_column_vals:
        curr_df = csv_df.loc[csv_df[group_by] == val]
        temp_dict = {}
        for column in csv_df_columns:
            temp_dict[column] = curr_df[column]
        ts_df = ts_df.append(temp_dict, ignore_index=True)
    return ts_df, csv_group_by_column_vals


def write_df_to_data(ts_labels, df, out_path, dimensions):
    """
    Writes df to data in .ts format

    Parameters
    ----------
    ts_labels: dict
        Dict containing the header labels for the ts format
    df: dataframe
        Dataframe containing the data to be written to .ts format
    out_path: str
        String path to output files
    """
    file_data = ""
    for key in ts_labels:
        file_data += "@" + str(key) + " " + str(ts_labels[key]) + "\n"
    file_data += "@data"
    lines = []
    class_label = []
    for key in df:
        i = 0
        df_dict = df[key].to_dict()
        for dict_key in df_dict:
            curr_series = df_dict[dict_key]
            if type(curr_series) != pd.Series:
                class_label.append(str(curr_series))
            else:
                curr_arr = curr_series.to_list()
                curr_str_vals = ""
                for val in curr_arr:
                    curr_str_vals += str(val)
                if i >= len(lines):
                    lines.append(curr_str_vals)
                else:
                    lines[i] += ":" + curr_str_vals
            i += 1
    for i in range(0, len(lines)):
        lines[i] += ":" + class_label[i]
        file_data += "\n" + lines[i]
    if ".ts" not in out_path:
        out_path += ".ts"
    with open(out_path, "w") as file:
        file.write(file_data)
    return file_data


def csv_to_ts_format(
    problem_name,
    path_to_csv,
    out_path,
    group_by,
    class_label_column=None,
    timestamps="false",
):
    """
    Loads data from a csv and writes out the ts format of that data. This
    includes setting correct headers

    Parameters
    ----------
    problem_name: str
        The desired name for the dataset
    path_to_csv: str
        The full pathname for the .csv file to read
    out_path: str
        The full pathname for the .ts file to be written to
    group_by: str
        The name of the column to group by
    class_label_column: int
        Integer that is the index of the class label column
    timestamps: boolean
        Optional parameter defaulting to False that states if the data has a
        timestamp
    return_headers: boolean
        Optional parameter defaulting to False that returns the headers set
        (mainly used for testing)
    """
    csv_df = pd.read_csv(path_to_csv)
    df, csv_group_by_column_vals = csv_to_ts_df(csv_df, group_by)

    dimensions = len(df.columns)

    columns = df.columns
    equal_length = "true"
    seires_length = 0
    for column in columns:
        previous_length = len(df[column][0])
        for val in df[column]:
            curr_length = len(val)
            if curr_length != previous_length:
                equal_length = "false"
            if curr_length > seires_length:
                seires_length = curr_length

    class_label = "false"
    if class_label_column is not None:
        if class_label_column in df:
            class_labels_vals = df[class_label_column]
            class_label_arr = []
            for val in class_labels_vals:
                class_label_arr.append(val[0])
            df["label"] = class_label_arr
            df = df.drop([class_label_column], axis=1)
        else:
            df["label"] = csv_group_by_column_vals

        class_labels_vals = set(df["label"])
        class_label = "true"
        for val in class_labels_vals:
            class_label += " " + str(val)

    univariate = "true"
    if dimensions > 1:
        univariate = "false"

    missing = "false"
    regex_str = r"^\?$"
    for col in df.columns:
        if col == "label":
            continue
        for arr in df[col]:
            for val in arr:
                val = str(val)
                if re.match(regex_str, val) is not None or val == "nan":
                    missing = "true"
                    break
            if missing == "true":
                break
        if missing == "true":
            break

    ts_label = {
        "problamName": problem_name,
        "timestamps": timestamps,
        "missing": missing,
        "univariate": univariate,
        "equalLength": equal_length,
        "seriesLength": seires_length,
        "dimensions": dimensions,
        "classLabel": class_label,
    }
    write_df_to_data(ts_label, df, out_path, dimensions)


def arff_to_ts_format(path_to_csv, out_path):
    """
    Loads data from a arff and writes out the ts format of that data. This
    includes setting correct headers

    Parameters
    ----------
    path_to_csv: str
        The full pathname for the .csv file to read
    out_path: str
        The full pathname for the .ts file to be written to
    """


def txt_to_ts_format(path_to_csv, out_path):
    """
    Loads data from a txt and writes out the ts format of that data. This
    includes setting correct headers

    Parameters
    ----------
    path_to_csv: str
        The full pathname for the .csv file to read
    out_path: str
        The full pathname for the .ts file to be written to
    """
