"""Util function for writing to and reading arff files."""

__author__ = ["SebasKoel", "Emiliathewolf", "TonyBagnall", "jasonlines", "achieveordie"]
__all__ = ["load_from_arff_to_dataframe"]

import itertools
import os
import textwrap

import numpy as np
import pandas as pd

from sktime.datasets._readers_writers.utils import get_path
from sktime.transformations.base import BaseTransformer

# ========================================================================
# Utils function to read  arff file
# ========================================================================


# TODO: original author didn't add test for this function
# Refactor the nested loops
def load_from_arff_to_dataframe(
    full_file_path_and_name,
    has_class_labels=True,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .arff file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .arff file to read.
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

    full_file_path_and_name = get_path(full_file_path_and_name, ".arff")

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, encoding="utf-8") as f:
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


# ================================================================================
# Utils function to write results from tabular transformation to arff file
# ================================================================================


# Research function?
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
    """Transform dataset using a tabular transformer and write the result to arff file.

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
    except OSError:
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
