"""Util functions for reading and writing .ts files."""

__author__ = ["Emiliathewolf", "TonyBagnall", "jasonlines", "achieveordie"]

__all__ = [
    "load_from_tsfile_to_dataframe",
    "load_from_tsfile",
    "write_dataframe_to_tsfile",
    "write_ndarray_to_tsfile",
    "write_panel_to_tsfile",
]

import itertools

import numpy as np
import pandas as pd

from sktime.datasets._readers_writers.utils import (
    _alias_mtype_check,
    _write_header,
    get_path,
)
from sktime.datatypes import MTYPE_LIST_PANEL, check_is_scitype, convert, convert_to
from sktime.utils.validation.panel import check_X, check_X_y

# ===================================================================================
# Function to read  .ts file
# ===================================================================================


# TODO: Highle deeply nested function, refactor
def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
    encoding="utf-8",
    y_dtype="str",
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
    encoding: str
        encoding is the name of the encoding used to read file using the open function.

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

    full_file_path_and_name = get_path(full_file_path_and_name, ".ts")

    # Parse the file
    with open(full_file_path_and_name, encoding=encoding) as file:
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
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise OSError(
                            "timestamps tag requires an associated Boolean value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise OSError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise OSError(
                            "univariate tag requires an associated Boolean  value"
                        )
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise OSError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError(
                            "classlabel tag requires an associated Boolean  value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise OSError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise OSError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    if data_started:
                        raise OSError("metadata must come before data")
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError(
                            "targetlabel tag requires an associated Boolean value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise OSError("invalid targetlabel value")
                    if token_len > 2:
                        raise OSError(
                            "targetlabel tag should not be accompanied with info "
                            "apart from true/false, but found "
                            f"{tokens}"
                        )
                    has_class_labels_tag = True
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise OSError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise OSError("metadata must come before data")
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
                        raise OSError(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we are dealing with data that has timestamps
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
                                            raise OSError(
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
                                            raise OSError(
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
                                            raise OSError(
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
                                            raise OSError(
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
                                            raise OSError(
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
                                            raise OSError(
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
                                            raise OSError(
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
                                            raise OSError(
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
                                raise OSError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise OSError(
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
                                    raise OSError(
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
                            raise OSError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise OSError(
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
                            raise OSError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise OSError("the cases have no associated class values")
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
                            raise OSError(
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
            raise OSError("metadata incomplete")

        elif metadata_started and not data_started:
            raise OSError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise OSError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data_dict = {
            f"dim_{dim}": pd.Series(instance_list[dim]) for dim in range(num_dimensions)
        }
        data = pd.DataFrame(data_dict)

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
        raise OSError("empty file")


def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_y=True,
    return_data_type="nested_univ",
    encoding="utf-8",
    y_dtype="str",
):
    """Load time series .ts file into X and (optionally) y.

    Data from a .ts file is loaded into a nested pd.DataFrame, or optionally into a
    2d np.ndarray (equal length, univariate problem) or 3d np.ndarray (equal length,
    multivariate problem) if requested. If present, y is loaded into a 1d np.ndarray.

    Parameters
    ----------
    full_file_path_and_name : str
        The full pathname and file name of the .ts file to read.
    replace_missing_vals_with : str, default NaN
       The value that missing values in the text file should be replaced with prior
       to parsing.
    return_y : boolean, default True
       whether to return the y variable, if it is present.
    return_data_type : str, optional, default = "nested_univ"
        memory data format specification to return X in.
        str can be any other supported Panel mtype
            for list of mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            "nested_univ: nested pd.DataFrame, pd.Series in cells
            "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.
    encoding: str
        encoding is the name of the encoding used to read file using the open function.
    y_dtype : str, optional, default 'float'
        cast y to this dtype before returning, accepted arguments are
        'str', 'int', and 'float'.

    Returns
    -------
    X : sktime compatible in-memory container of mtype return_data_type
        for list of mtypes, see datatypes.SCITYPE_REGISTER
        for specifications, see examples/AA_datatypes_and_datasets.ipynb
    y : returned only if return_y=True, np.ndarray

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
    return_data_type = _alias_mtype_check(return_data_type)

    if not isinstance(return_data_type, str):
        raise TypeError(
            f"return_data_type argument must be a str, but found "
            f"{type(return_data_type)}"
        )
    if return_data_type not in MTYPE_LIST_PANEL:
        raise ValueError(
            f"return_data_type must be one of the following identifier strings for "
            f"sktime panel time series data format specifications: {MTYPE_LIST_PANEL}, "
            f"but found {return_data_type}"
        )

    # Initialize flags and variables used when parsing the file
    X, y = load_from_tsfile_to_dataframe(
        full_file_path_and_name=full_file_path_and_name,
        return_separate_X_and_y=True,
        replace_missing_vals_with=replace_missing_vals_with,
        encoding=encoding,
        y_dtype=y_dtype,
    )
    if isinstance(y_dtype, np.dtype):
        y = y.astype(y_dtype)
    elif y_dtype == "str":
        y = y.astype(str)
    elif y_dtype == "int":
        y = y.astype(np.int8)
    elif y_dtype == "float":
        y = y.astype(np.float32)
    else:
        raise ValueError(f"Invalid y_dtype: {y_dtype}")
    X = convert(X, from_type="nested_univ", to_type=return_data_type)

    if return_y:
        return X, y
    else:
        return X


# =============================================================================
# Functions to write data to .ts format.
# =============================================================================


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
    """Output a dataset in dataframe format to .ts file.

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
        raise ValueError(f"Data provided must be a DataFrame, passed a {type(data)}")
    data_valid, _, metadata = check_is_scitype(
        data, scitype="Panel", return_metadata=True
    )
    if not data_valid:
        raise ValueError("DataFrame provided is not a valid type")
    if equal_length != metadata["is_equal_length"]:
        raise ValueError(
            f"Argument passed for equal length = {equal_length} is not "
            f"true for the data passed"
        )
    if equal_length:
        # Convert to [cases][dimensions][length] numpy.
        data = convert_to(
            data,
            to_type="numpy3D",
            as_scitype="Panel",
            store_behaviour="freeze",
        )
        write_ndarray_to_tsfile(
            data,
            path,
            problem_name=problem_name,
            class_label=class_label,
            class_value_list=class_value_list,
            equal_length=equal_length,
            series_length=data.shape[2],
            missing_values=missing_values,
            comment=comment,
            fold=fold,
        )
    else:  # Write by iterating over dataframe
        if class_value_list is not None and class_label is None:
            class_label = np.unique(class_value_list)
        file = _write_header(
            path,
            problem_name,
            metadata["is_univariate"],
            metadata["is_equal_length"],
            series_length,
            class_label,
            fold,
            comment,
        )
        n_cases, n_dimensions = data.shape
        for i in range(0, n_cases):
            for j in range(0, n_dimensions):
                series = data.iloc[i, j]
                for k in range(0, series.size - 1):
                    file.write(f"{series[k]},")
                file.write(f"{series[series.size - 1]}:")
            file.write(f"{class_value_list[i]}\n")


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
    """Output a dataset in ndarray format to .ts file.

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
    file = _write_header(
        path,
        problem_name,
        univariate,
        equal_length,
        series_length,
        class_label,
        fold,
        comment,
    )
    # begin writing the core data for each case
    # which are the series and the class value list if there is any
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


def write_panel_to_tsfile(
    data, path, target=None, problem_name="sample_data", header=None
):
    """Write an sktime multi-instance dataset to text file in .ts format.

    Write metadata and data stored in sktime compatible data set to file.
    A description of the ts format is in docs/source/api_reference/data_format.rst

    Parameters
    ----------
    data : Panel.
        dataset containing multiple time series instances, referred to as a Panel in
        sktime.
        Series can univariate, multivariate, equal or unequal length
    path : String.
        Location of the directory to write file
    target: None or ndarray, default = None
        Response variable, discrete for classification, continuous for regression
        None if clustering.
    problem_name : String, default = "sample_data"
        The file is written to <path>/<problem_name>/<problem_name>.ts
    header: String, default = None
        Optional text at the top of the file that is ignored when loading.
    """
    data_valid, _, data_metadata = check_is_scitype(
        data, scitype="Panel", return_metadata=True
    )
    if not data_valid:
        raise TypeError(" Wrong input data type ", type(data))
    if data_metadata["is_equal_length"]:
        # check class labels
        data = convert_to(
            data,
            to_type="numpy3D",
            as_scitype="Panel",
            store_behaviour="freeze",
        )
        series_length = data.shape[2]
        write_ndarray_to_tsfile(
            data,
            path,
            problem_name=problem_name,
            class_value_list=target,
            equal_length=True,
            series_length=series_length,
            comment=header,
        )
    else:
        write_dataframe_to_tsfile(
            data,
            path,
            problem_name=problem_name,
            class_value_list=target,
            equal_length=False,
            comment=header,
        )
