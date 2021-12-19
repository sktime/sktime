# -*- coding: utf-8 -*-
"""Functions for the input and output of data and results.

todo: This file will be removed in version 0.10 and functionality moved to
datasets/_data_io.py
"""

import itertools
import os
import textwrap
from warnings import warn

import numpy as np
import pandas as pd

from sktime.datatypes._panel._convert import _make_column_names, from_long_to_nested
from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.panel import check_X, check_X_y


class TsFileParseException(Exception):
    """Should be raised when parsing a .ts file and the format is incorrect."""

    pass


class LongFormatDataParseException(Exception):
    """Should be raised when parsing a .csv file with long-formatted data."""

    pass


def load_from_arff_to_dataframe(
    full_file_path_and_name,
    has_class_labels=True,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    has_class_labels: bool
        true then line contains separated strings and class value contains
        list of separated strings, check for 'return_separate_X_and_y'
        false otherwise.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    instance_list = []
    class_val_list = []

    data_started = False
    is_multi_variate = False
    is_first_case = True

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, "r", encoding="utf-8") as f:
        for line in f:

            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for _d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(
                                pd.Series(
                                    [float(i) for i in dimensions[dim].split(",")]
                                )
                            )

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(
                                pd.Series(
                                    [
                                        float(i)
                                        for i in line_parts[: len(line_parts) - 1]
                                    ]
                                )
                            )
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(
                                pd.Series(
                                    [float(i) for i in line_parts[: len(line_parts)]]
                                )
                            )

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data["class_vals"] = pd.Series(class_val_list)

    return x_data


def load_from_ucr_tsv_to_dataframe(
    full_file_path_and_name, return_separate_X_and_y=True
):
    """Load data from a .tsv file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .tsv file to read.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    df = pd.read_csv(full_file_path_and_name, sep="\t", header=None)
    y = df.pop(0).values
    df.columns -= 1
    X = pd.DataFrame()
    X["dim_0"] = [pd.Series(df.iloc[x, :]) for x in range(len(df))]
    if return_separate_X_and_y is True:
        return X, y
    X["class_val"] = y
    return X


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
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    data = pd.read_csv(full_file_path_and_name, sep=separator, header=0)
    # ensure there are 4 columns in the long_format table
    if len(data.columns) != 4:
        raise LongFormatDataParseException("dataframe must contain 4 columns of data")

    # ensure that all columns contain the correct data types
    if (
        not data.iloc[:, 0].dtype == "int64"
        or not data.iloc[:, 1].dtype == "int64"
        or not data.iloc[:, 2].dtype == "int64"
        or not data.iloc[:, 3].dtype == "float64"
    ):
        raise LongFormatDataParseException(
            "one or more data columns contains data of an incorrect type"
        )

    data = from_long_to_nested(data)
    return data


# left here for now, better elsewhere later perhaps
def generate_example_long_table(num_cases=50, series_len=20, num_dims=2):
    """Generate example from long table format file.

    Parameters
    ----------
    num_cases: int
        Number of cases.
    series_len: int
        Length of the series.
    num_dims: int
        Number of dimensions.

    Returns
    -------
    DataFrame
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    rows_per_case = series_len * num_dims
    total_rows = num_cases * series_len * num_dims

    case_ids = np.empty(total_rows, dtype=int)
    idxs = np.empty(total_rows, dtype=int)
    dims = np.empty(total_rows, dtype=int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_len)
        idxs[i] = rem % series_len

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


def make_multi_index_dataframe(n_instances=50, n_columns=3, n_timepoints=20):
    """Generate example multi-index DataFrame.

    Parameters
    ----------
    n_instances : int
        Number of instances.
    n_columns : int
        Number of columns (series) in multi-indexed DataFrame.
    n_timepoints : int
        Number of timepoints per instance-column pair.

    Returns
    -------
    mi_df : pd.DataFrame
        The multi-indexed DataFrame with
        shape (n_instances*n_timepoints, n_column).
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    # Make long DataFrame
    long_df = generate_example_long_table(
        num_cases=n_instances, series_len=n_timepoints, num_dims=n_columns
    )
    # Make Multi index DataFrame
    mi_df = long_df.set_index(["case_id", "reading_id"]).pivot(columns="dim_id")
    mi_df.columns = _make_column_names(n_columns)

    return mi_df


def write_results_to_uea_format(
    estimator_name,
    dataset_name,
    y_pred,
    output_path,
    full_path=True,
    y_true=None,
    predicted_probs=None,
    split="TEST",
    resample_seed=0,
    timing_type="N/A",
    first_line_comment=None,
    second_line="No Parameter Info",
    third_line="N/A",
):
    """Write the predictions for an experiment in the standard format used by sktime.

    Parameters
    ----------
    estimator_name : str,
        Name of the object that made the predictions, written to file and can
        deterimine file structure of output_root is True
    dataset_name : str
        name of the problem the classifier was built on
    y_pred : np.array
        predicted values
    output_path : str
        Path where to put results. Either a root path, or a full path
    full_path : boolean, default = True
        If False, then the standard file structure is created. If false, results are
        written directly to the directory passed in output_path
    y_true : np.array, default = None
        Actual values, written to file with the predicted values if present
    predicted_probs :  np.ndarray, default = None
        Estimated class probabilities. If passed, these are written after the
        predicted values. Regressors should not pass anything
    split : str, default = "TEST"
        Either TRAIN or TEST, depending on the results, influences file name.
    resample_seed : int, default = 0
        Indicates what data
    timing_type : str or None, default = None
        The format used for timings in the file, i.e. Seconds, Milliseconds, Nanoseconds
    first_line_comment : str or None, default = None
        Optional comment appended to the end of the first line
    second_line : str
        unstructured, used for predictor parameters
    third_line : str
        summary performance information (see comment below)
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    if len(y_true) != len(y_pred):
        raise IndexError(
            "The number of predicted values is not the same as the "
            "number of actual class values"
        )

    # If the full directory path is not passed, make the standard structure
    if not full_path:
        output_path = (
            str(output_path)
            + "/"
            + str(estimator_name)
            + "/Predictions/"
            + str(dataset_name)
            + "/"
        )
    try:
        os.makedirs(output_path)
    except os.error:
        pass  # raises os.error if path already exists, so just ignore this

    if split == "TRAIN" or split == "train":
        train_or_test = "train"
    elif split == "TEST" or split == "test":
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")

    file = open(
        str(output_path)
        + "/"
        + str(train_or_test)
        + "Resample"
        + str(resample_seed)
        + ".csv",
        "w",
    )

    # the first line of the output file is in the form of:
    # <classifierName>,<datasetName>,<train/test>
    first_line = (
        str(estimator_name) + "," + str(dataset_name) + "," + str(train_or_test)
    )
    if timing_type is not None:
        first_line += "," + timing_type
    if first_line_comment is not None:
        first_line += "," + first_line_comment
    file.write(first_line + "\n")

    # the second line of the output is free form and estimator-specific; usually this
    # will record info such as build time, paramater options used, any constituent model
    # names for ensembles, etc.
    file.write(str(second_line) + "\n")

    # the third line of the file is the accuracy (should be between 0 and 1
    # inclusive). If this is a train output file then it will be a training estimate
    # of the classifier on the training data only (e.g. 10-fold cv, leave-one-out cv,
    # etc.). If this is a test output file, it should be the output of the estimator
    # on the test data (likely trained on the training data for a-priori parameter
    # optimisation)
    file.write(str(third_line) + "\n")

    # from line 4 onwards each line should include the actual and predicted class
    # labels (comma-separated). If present, for each case, the probabilities of
    # predicting every class value for this case should also be appended to the line (
    # a space is also included between the predicted value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   y_true[i], y_pred[i],,prob_class_0[i],
    #   prob_class_1[i],...,prob_class_c[i]
    #
    # if predict_proba data IS NOT provided for case i:
    #   y_true[i], y_pred[i]
    # If y_true is None (if clustering), y_true[i] is replaced with ? to indicate
    # missing
    if y_true is None:
        for i in range(0, len(y_pred)):
            file.write("?," + str(y_pred[i]))
            if predicted_probs is not None:
                file.write(",")
                for j in predicted_probs[i]:
                    file.write("," + str(j))
            file.write("\n")
    else:
        for i in range(0, len(y_pred)):
            file.write(str(y_true[i]) + "," + str(y_pred[i]))
            if predicted_probs is not None:
                file.write(",")
                for j in predicted_probs[i]:
                    file.write("," + str(j))
            file.write("\n")
    file.close()


def write_tabular_transformation_to_arff(
    data,
    transformation,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    comment=None,
    fold="",
    fit_transform=True,
):
    """
    Transform a dataset using a tabular transformer and write the result to a arff file.

    Parameters
    ----------
    data: pandas dataframe or 3d numpy array
        The dataset to build the transformation with which must be of the structure
        specified in the documentation examples/loading_data.ipynb.
    transformation: BaseTransformer
        Transformation use and to save to arff.
    path: str
        The full path to output the arff file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the arff file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header, optional.
    class_value_list: list, ndarray or None, default=None
        The class values for each case, optional.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.
    fit_transform: bool, default=True
        Whether to fit the transformer prior to calling transform.

    Returns
    -------
    None
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10",
        FutureWarning,
    )
    # ensure transformation provided is a transformer
    if not isinstance(transformation, BaseTransformer):
        raise ValueError("Transformation must be a BaseTransformer")

    if fit_transform:
        data = transformation.fit_transform(data, class_value_list)
    else:
        data = transformation.transform(data, class_value_list)

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    if class_value_list is not None and class_label is None:
        class_label = np.unique(class_value_list)
    elif class_value_list is None:
        class_value_list = []

    # ensure number of cases is same as the class value list
    if len(data) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the number of given class values"
        )

    if fold is None:
        fold = ""

    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}-{type(transformation).__name__}/"
    try:
        os.makedirs(dirt)
    except os.error:
        pass  # raises os.error if path already exists

    # create arff file in the path
    file = open(
        f"{dirt}{str(problem_name)}-{type(transformation).__name__}{fold}.arff", "w"
    )

    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n% ".join(textwrap.wrap("% " + comment)))
        file.write("\n")

    # begin writing header information
    file.write(f"@Relation {problem_name}\n")

    # write each attribute
    for i in range(data.shape[1]):
        file.write(f"@attribute att{str(i)} numeric\n")

    # write class attribute if it exists
    if class_label is not None:
        comma_separated_class_label = ",".join(str(label) for label in class_label)
        file.write(f"@attribute target {{{comma_separated_class_label}}}\n")

    file.write("@data\n")

    for case, value in itertools.zip_longest(data, class_value_list):
        # turn attributes into comma-separated row
        atts = ",".join([str(num) if not np.isnan(num) else "?" for num in case])
        file.write(str(atts))
        if value is not None:
            file.write(f",{value}")  # write the case value if any
        elif class_label is not None:
            file.write(",?")
        file.write("\n")  # open a new line

    file.close()


def write_dataframe_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
    fold="",
):
    """
    Output a dataset in dataframe format to .ts file.

    Parameters
    ----------
    data: pandas dataframe
        The dataset in a dataframe to be written as a ts file
        which must be of the structure specified in the documentation
        examples/loading_data.ipynb.
        index |   dim_0   |   dim_1   |    ...    |  dim_c-1
           0  | pd.Series | pd.Series | pd.Series | pd.Series
           1  | pd.Series | pd.Series | pd.Series | pd.Series
          ... |    ...    |    ...    |    ...    |    ...
           n  | pd.Series | pd.Series | pd.Series | pd.Series
    path: str
        The full path to output the ts file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the ts file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header, optional.
    class_value_list: list, ndarray or None, default=None
        The class values for each case, optional.
    equal_length: bool, default=False
        Indicates whether each series is of equal length.
    series_length: int, default=-1
        Indicates the series length if they are of equal length.
    missing_values: str, default="NaN"
        Representation for missing values.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10"
    )
    # ensure data provided is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data provided must be a DataFrame")

    if class_value_list is not None:
        data, class_value_list = check_X_y(data, class_value_list, coerce_to_numpy=True)
    else:
        data = check_X(data, coerce_to_numpy=True)

    # ensure data provided is a dataframe
    write_ndarray_to_tsfile(
        data,
        path,
        problem_name=problem_name,
        class_label=class_label,
        class_value_list=class_value_list,
        equal_length=equal_length,
        series_length=series_length,
        missing_values=missing_values,
        comment=comment,
        fold=fold,
    )


def write_ndarray_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
    fold="",
):
    """
    Output a dataset in ndarray format to .ts file.

    Parameters
    ----------
    data: pandas dataframe
        The dataset in a 3d ndarray to be written as a ts file
        which must be of the structure specified in the documentation
        examples/loading_data.ipynb.
        (n_instances, n_columns, n_timepoints)
    path: str
        The full path to output the ts file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the ts file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header.
    class_value_list: list, ndarray or None, default=None
        The class values for each case, optional.
    equal_length: bool, default=False
        Indicates whether each series is of equal length.
    series_length: int, default=-1
        Indicates the series length if they are of equal length.
    missing_values: str, default="NaN"
        Representation for missing values.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.
    """
    warn(
        "This function has moved to datasets/_data_io, this version will be removed "
        "in V0.10"
    )
    # ensure data provided is a ndarray
    if not isinstance(data, np.ndarray):
        raise ValueError("Data provided must be a ndarray")

    if class_value_list is not None:
        data, class_value_list = check_X_y(data, class_value_list)
    else:
        data = check_X(data)

    univariate = data.shape[1] == 1

    if class_value_list is not None and class_label is None:
        class_label = np.unique(class_value_list)
    elif class_value_list is None:
        class_value_list = []

    # ensure number of cases is same as the class value list
    if len(data) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the number of given class values"
        )

    if equal_length and series_length == -1:
        raise ValueError(
            "Please specify the series length for equal length time series data."
        )

    if fold is None:
        fold = ""

    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}/"
    try:
        os.makedirs(dirt)
    except os.error:
        pass  # raises os.error if path already exists

    # create ts file in the path
    file = open(f"{dirt}{str(problem_name)}{fold}.ts", "w")

    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")

    # begin writing header information
    file.write(f"@problemName {problem_name}\n")
    file.write("@timestamps false\n")
    file.write(f"@univariate {str(univariate).lower()}\n")
    file.write(f"@equalLength {str(equal_length).lower()}\n")

    if series_length > 0 and equal_length:
        file.write(f"@seriesLength {series_length}\n")

    # write class label line
    if class_label is not None:
        space_separated_class_label = " ".join(str(label) for label in class_label)
        file.write(f"@classLabel true {space_separated_class_label}\n")
    else:
        file.write("@class_label false\n")

    # begin writing the core data for each case
    # which are the series and the class value list if there is any
    file.write("@data\n")
    for case, value in itertools.zip_longest(data, class_value_list):
        for dimension in case:
            # turn series into comma-separated row
            series = ",".join(
                [str(num) if not np.isnan(num) else missing_values for num in dimension]
            )
            file.write(str(series))
            # continue with another dimension for multivariate case
            if not univariate:
                file.write(":")
        a = ":" if univariate else ""
        if value is not None:
            file.write(f"{a}{value}")  # write the case value if any
        elif class_label is not None:
            file.write(f"{a}{missing_values}")
        file.write("\n")  # open a new line

    file.close()
