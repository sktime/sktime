# -*- coding: utf-8 -*-
"""Utilities for loading datasets that are unapproved of in the main repo."""

__author__ = [
    "Emiliathewolf",
    "TonyBagnall",
    "jasonlines",
]

import numpy as np
import pandas as pd

from sktime.datasets._data_io import _read_header
from sktime.datatypes._panel._convert import (
    _make_column_names,
    from_long_to_nested,
    from_nested_to_2d_np_array,
    from_nested_to_3d_numpy,
)


def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_y=True,
):
    """Load time series data into X and (optionally) y.

    Data from a .ts file into a an 2D (univariate) or 3D (multivariate) if equal
    length or Pandas DataFrame if unequal length. If present, y is loaded into a 1D
    array.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    replace_missing_vals_with: str, default NaN
       The value that missing values in the text file should be replaced
       with prior to parsing.
    return_y: boolean default True
       whether to return the y variable, if it is present.

    Returns
    -------
    X: DataFrame or ndarray
    y (optional): ndarray.
    """
    # Initialize flags and variables used when parsing the file
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    num_dimensions = 0
    num_cases = 0
    # equal_length = True
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
        if not _meta_data["has_timestamps"] and _meta_data["is_equal_length"]:
            if _meta_data["is_univariate"]:
                data = from_nested_to_2d_np_array(data)
            else:
                data = from_nested_to_3d_numpy(data)
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
