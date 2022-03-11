# -*- coding: utf-8 -*-
"""Utilities for loading datasets."""

__author__ = [
    "Emiliathewolf",
    "TonyBagnall",
    "jasonlines",
]

__all__ = [
    "generate_example_long_table",
    "make_multi_index_dataframe",
    "write_dataframe_to_tsfile",
    "write_ndarray_to_tsfile",
    "write_results_to_uea_format",
    "write_tabular_transformation_to_arff",
    "load_from_tsfile",
    "load_from_tsfile_to_dataframe",
    "load_from_arff_to_dataframe",
    "load_from_long_to_dataframe",
    "load_from_ucr_tsv_to_dataframe",
]

import itertools
import os
import shutil
import tempfile
import textwrap
import zipfile
from datetime import datetime
from distutils.util import strtobool
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from sktime.datasets._data_io import MODULE
from sktime.datatypes._panel._convert import (
    _make_column_names,
    from_long_to_nested,
    from_nested_to_2d_np_array,
    from_nested_to_3d_numpy,
)
from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.panel import check_X, check_X_y

DIRNAME = "data"


# time series classification data sets
def _download_and_extract(url, extract_path=None):
    """
    Download and unzip datasets (helper function).

    This code was modified from
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L28

    Parameters
    ----------
    url : string
        Url pointing to file to download
    extract_path : string, optional (default: None)
        path to extract downloaded zip to, None defaults
        to sktime/datasets/data

    Returns
    -------
    extract_path : string or None
        if successful, string containing the path of the extracted file, None
        if it wasn't succesful

    """
    file_name = os.path.basename(url)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = os.path.join(dl_dir, file_name)
    urlretrieve(url, zip_file_name)

    if extract_path is None:
        extract_path = os.path.join(MODULE, "local_data/%s/" % file_name.split(".")[0])
    else:
        extract_path = os.path.join(extract_path, "%s/" % file_name.split(".")[0])

    try:
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        zipfile.ZipFile(zip_file_name, "r").extractall(extract_path)
        shutil.rmtree(dl_dir)
        return extract_path
    except zipfile.BadZipFile:
        shutil.rmtree(dl_dir)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        raise zipfile.BadZipFile(
            "Could not unzip dataset. Please make sure the URL is valid."
        )


def _list_available_datasets(extract_path=None):
    """Return a list of all the currently downloaded datasets.

    Modified version of
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L250

    Parameters
    ----------
    extract_path: (optional) None or string, location to look for data sest

    Returns
    -------
    datasets : List
        List of the names of datasets in the extract_path

    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, "data")
    else:
        data_dir = extract_path
    datasets = [
        path
        for path in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, path))
    ]
    return datasets


def _load_dataset(name, split, return_X_y, extract_path=None, return_type=None):
    """Load time series classification datasets."""
    # Allow user to have non standard extract path
    if extract_path is not None:
        path = os.path.join(extract_path)
    else:
        if name in _list_available_datasets():
            path = os.path.join(MODULE + "/data")
        else:
            path = os.path.join(MODULE + "/local_data")
    if not os.path.exists(path):
        os.makedirs(path)
    if name not in _list_available_datasets(path):
        # Dataset is not baked in the datasets directory, look in local_data,
        # if it is not there, download and install it.
        url = "http://timeseriesclassification.com/Downloads/%s.zip" % name
        # This also tests the validitiy of the URL, can't rely on the html
        # status code as it always returns 200
        try:
            _download_and_extract(
                url,
                extract_path=path,
            )
        except zipfile.BadZipFile as e:
            raise ValueError(
                "Invalid dataset name. ",
                extract_path,
                "Please make sure the dataset "
                + "is available on http://timeseriesclassification.com/.",
            ) from e
    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(path, name, fname)
        X, y = load_from_tsfile(abspath, return_data_type=return_type)
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        abspath = os.path.join(path, name, fname)
        X_train, y_train = load_from_tsfile(abspath, return_data_type=return_type)
        fname = name + "_TEST.ts"
        abspath = os.path.join(path, name, fname)
        X_test, y_test = load_from_tsfile(abspath, return_data_type=return_type)
        if isinstance(X_train, np.ndarray):
            X = np.concatenate((X_train, X_test))
        elif isinstance(X_train, pd.DataFrame):
            X = pd.concat([X_train, X_test])
        else:
            raise IOError(
                f"Invalid data structure type {type(X_train)} for loading "
                f"classification problem "
            )
        y = np.concatenate((y_train, y_test))
    else:
        raise ValueError("Invalid `split` value =", split)

    # Return appropriately
    if return_X_y:
        return X, y
    else:
        if isinstance(X, pd.DataFrame):
            X["class_val"] = pd.Series(y)
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                np.concatenate((X, y), axis=1)
            else:
                raise ValueError(
                    f"Unable to return multivariate X and y in {X.ndim} "
                    f"a numpy array"
                )
        return X


def _load_provided_dataset(name, split=None, return_X_y=True, return_type=None):
    """Load baked in time series classification datasets (helper function)."""
    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(MODULE, DIRNAME, name, fname)
        X, y = load_from_tsfile(abspath, return_data_type=return_type)
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        abspath = os.path.join(MODULE, DIRNAME, name, fname)
        X_train, y_train = load_from_tsfile(abspath, return_data_type=return_type)
        fname = name + "_TEST.ts"
        abspath = os.path.join(MODULE, DIRNAME, name, fname)
        X_test, y_test = load_from_tsfile(abspath, return_data_type=return_type)
        if isinstance(X_train, np.ndarray):
            X = np.concatenate((X_train, X_test))
        elif isinstance(X_train, pd.DataFrame):
            X = pd.concat([X_train, X_test])
        else:
            raise IOError(
                f"Invalid data structure type {type(X_train)} for loading "
                f"classification problem "
            )
        y = np.concatenate((y_train, y_test))

    else:
        raise ValueError("Invalid `split` value =", split)
    # Return appropriately
    if return_X_y:
        return X, y
    else:
        if isinstance(X, pd.DataFrame):
            X["class_val"] = pd.Series(y)
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                np.concatenate((X, y), axis=1)
            else:
                raise ValueError(
                    f"Unable to return multivariate X and y in {X.ndim} "
                    f"a numpy array"
                )
        return X


def _read_header(file, full_file_path_and_name):
    """Read the header information, returning the meta information."""
    # Meta data for data information
    meta_data = {
        "is_univariate": True,
        "is_equally_spaced": True,
        "is_equal_length": True,
        "has_nans": False,
        "has_timestamps": False,
        "has_class_labels": True,
    }
    # Read header until @data tag met
    for line in file:
        line = line.strip().lower()
        if line:
            if line.startswith("@problemname"):
                tokens = line.split(" ")
                token_len = len(tokens)
            elif line.startswith("@timestamps"):
                tokens = line.split(" ")
                if tokens[1] == "true":
                    meta_data["has_timestamps"] = True
                elif tokens[1] != "false":
                    raise IOError(
                        f"invalid timestamps tag value {tokens[1]} value in file "
                        f"{full_file_path_and_name}"
                    )
            elif line.startswith("@univariate"):
                tokens = line.split(" ")
                token_len = len(tokens)
                if tokens[1] == "false":
                    meta_data["is_univariate"] = False
                elif tokens[1] != "true":
                    raise IOError(
                        f"invalid univariate tag value {tokens[1]} in file "
                        f"{full_file_path_and_name}"
                    )
            elif line.startswith("@equallength"):
                tokens = line.split(" ")
                if tokens[1] == "false":
                    meta_data["is_equal_length"] = False
                elif tokens[1] != "true":
                    raise IOError(
                        f"invalid unequal tag value {tokens[1]} in file "
                        f"{full_file_path_and_name}"
                    )
            elif line.startswith("@classlabel"):
                tokens = line.split(" ")
                token_len = len(tokens)
                if tokens[1] == "false":
                    meta_data["class_labels"] = False
                elif tokens[1] != "true":
                    raise IOError(
                        "invalid classLabel value in file " f"{full_file_path_and_name}"
                    )
                if token_len == 2 and meta_data["class_labels"]:
                    raise IOError(
                        f"if the classlabel tag is true then class values must be "
                        f"supplied in file{full_file_path_and_name} but read {tokens}"
                    )
            elif line.startswith("@data"):
                return meta_data
    raise IOError(
        f"End of file reached for {full_file_path_and_name} but no indicated start of "
        f"data with the tag @data"
    )


def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_y=True,
    return_data_type=None,
):
    """Load time series .ts file into X and (optionally) y.

    Data from a .ts file is loaded into a nested pd.DataFrame, or optionally into a
    2d np.ndarray (equal length, univariate problem) or 3d np.ndarray (eqal length,
    multivariate problem) if requested. If present, y is loaded into a 1d .

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname and file name of the .ts file to read.
    replace_missing_vals_with: str, default NaN
       The value that missing values in the text file should be replaced with prior
       to parsing.
    return_y: boolean, default True
       whether to return the y variable, if it is present.
    return_data_type: default, None
        what data structure to return X in. Valid alternatives to None (default to
        pd.DataFrame) are nested_univ (pd.DataFrame) are numpy2d/np2d/numpyflat or
        numpy3d/np3d.
        These arguments will raise an error if the data cannot be stored in the
        requested type.

    Returns
    -------
    X:  If return_data_type = None retuns X in a nested pd.dataframe
        If return_data_type = numpy3d/np3ddefault DataFrame or ndarray
    y (optional): ndarray.

    Raises
    ------
    IOError if the requested file does not exist
    IOError if input series are not all the same dimension (not supported)
    IOError if class labels have been requested but are not present in the file
    IOError if the input file has no cases
    ValueError if return_data_type = numpy3d but the data are unequal length series
    ValueError if return_data_type = numpy2d but the data are multivariate and/
    or unequal length series

    """
    # Initialize flags and variables used when parsing the file
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    num_dimensions = 0
    num_cases = 0
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        _meta_data = _read_header(file, full_file_path_and_name)
        for line in file:  # Will this work?
            num_cases = num_cases + 1
            line = line.replace("?", replace_missing_vals_with)
            dimensions = line.split(":")
            # If first instance then note the number of dimensions.
            # This must be the same for all cases.
            if is_first_case:
                num_dimensions = len(dimensions)
                if _meta_data["has_class_labels"]:
                    num_dimensions -= 1
                for _dim in range(0, num_dimensions):
                    instance_list.append([])
                is_first_case = False
                _meta_data["num_dimensions"] = num_dimensions
            # See how many dimensions a case has
            this_line_num_dim = len(dimensions)
            if _meta_data["has_class_labels"]:
                this_line_num_dim -= 1
            if this_line_num_dim != _meta_data["num_dimensions"]:
                raise IOError(
                    f"Error input {full_file_path_and_name} all cases must "
                    f"have the {num_dimensions} dimensions. Case "
                    f"{num_cases} has {this_line_num_dim}"
                )
            # Process the data for each dimension
            for dim in range(0, _meta_data["num_dimensions"]):
                dimension = dimensions[dim].strip()
                if dimension:
                    data_series = dimension.split(",")
                    data_series = [float(i) for i in data_series]
                    instance_list[dim].append(pd.Series(data_series))
                else:
                    instance_list[dim].append(pd.Series(dtype="object"))
            if _meta_data["has_class_labels"]:
                class_val_list.append(dimensions[_meta_data["num_dimensions"]].strip())
                line_num += 1
    # Check that the file was not empty
    if line_num:
        # Create a DataFrame from the data parsed
        data = pd.DataFrame(dtype=np.float32)
        for dim in range(0, _meta_data["num_dimensions"]):
            data["dim_" + str(dim)] = instance_list[dim]
        # convert to numpy if the user requests it.
        if isinstance(return_data_type, str):
            return_data_type = return_data_type.strip().lower()
        if (
            return_data_type == "numpy2d"
            or return_data_type == "np2d"
            or return_data_type == "numpyflat"
        ):
            if (
                not _meta_data["has_timestamps"]
                and _meta_data["is_equal_length"]
                and _meta_data["is_univariate"]
            ):
                data = from_nested_to_2d_np_array(data)
            else:
                raise ValueError(
                    "Unable to convert to 2d numpy as requested, "
                    "because at least one flag means the data structure "
                    f"cannot be used {_meta_data}"
                )
        elif return_data_type == "numpy3d" or return_data_type == "np3d":
            if not _meta_data["has_timestamps"] and _meta_data["is_equal_length"]:
                data = from_nested_to_3d_numpy(data)
            else:
                raise ValueError(
                    " Unable to convert to 3d numpy as requested, "
                    "because at least one flag means the data structure "
                    f"cannot be used, meta data = {_meta_data}"
                )
        if return_y and not _meta_data["has_class_labels"]:
            raise IOError(
                f"class labels have been requested, but they "
                f"are not present in the file "
                f"{full_file_path_and_name}"
            )
        if _meta_data["has_class_labels"] and return_y:
            return data, np.asarray(class_val_list)
        else:
            return data
    else:
        raise IOError(
            f"Empty file {full_file_path_and_name} with header info but no " f"cases"
        )


def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    # Parse the file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "timestamps tag requires an associated Boolean " "value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise IOError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "univariate tag requires an associated Boolean  " "value"
                        )
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise IOError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError(
                            "classlabel tag requires an associated Boolean  " "value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise IOError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise IOError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise IOError("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise IOError(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False
                        timestamp_for_dim = []
                        values_for_dimension = []
                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0
                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1
                            # See if there is any more data to read in or if
                            # we should validate that read thus far
                            if char_num < line_len:
                                # See if we have an empty dimension (i.e. no
                                # values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])
                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
                                    this_line_num_dim += 1
                                    has_another_value = False
                                    has_another_dimension = True
                                    timestamp_for_dim = []
                                    values_for_dimension = []
                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and class_labels:
                                        class_val = line[char_num:].strip()
                                        if class_val not in class_label_list:
                                            raise IOError(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                "valid"
                                            )
                                        class_val_list.append(class_val)
                                        char_num = line_len
                                        has_another_value = False
                                        has_another_dimension = False
                                        timestamp_for_dim = []
                                        values_for_dimension = []
                                    else:
                                        # Read in the data contained within
                                        # the next tuple
                                        if line[char_num] != "(" and not class_labels:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )
                                        char_num += 1
                                        tuple_data = ""
                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )
                                        # Read in any spaces immediately
                                        # after the current tuple
                                        char_num += 1
                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple
                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False
                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False
                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True
                                        char_num += 1
                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma
                                        last_comma_index = tuple_data.rfind(",")
                                        if last_comma_index == -1:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )
                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)
                                        except ValueError:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )
                                        # Check the type of timestamp that
                                        # we have
                                        timestamp = tuple_data[0:last_comma_index]
                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False
                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False
                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent
                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        # Store the values
                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]
                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had
                                        if (
                                            prev_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1
                                            timestamp_for_dim = []
                                            values_for_dimension = []
                            elif has_another_value:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise IOError(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )
                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])
                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dim
                        ):
                            raise IOError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise IOError("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)
                            if class_labels:
                                num_dimensions -= 1
                            for _dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False
                        # See how many dimensions that the case whose data
                        # in represented in this line has
                        this_line_num_dim = len(dimensions)
                        if class_labels:
                            this_line_num_dim -= 1
                        # All dimensions should be included for all series,
                        # even if they are empty
                        if this_line_num_dim != num_dimensions:
                            raise IOError(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
                            )
                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))
                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())
            line_num += 1
    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise IOError("metadata incomplete")

        elif metadata_started and not data_started:
            raise IOError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise IOError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)
        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]
        # Check if we should return any associated class labels separately
        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise IOError("empty file")


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


def load_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    """
    Convert the contents in a .tsf file into a dataframe.

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

    Returns
    -------
    loaded_data: pd.DataFrame
        The converted dataframe containing the time series.
    frequency: str
        The frequency of the dataset.
    forecast_horizon: int
        The expected forecast horizon of the dataset.
    contain_missing_values: bool
        Whether the dataset contains missing values or not.
    contain_equal_length: bool
        Whether the series have equal lengths or not.
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

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
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

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )
