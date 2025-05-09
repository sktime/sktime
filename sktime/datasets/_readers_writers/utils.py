"""Private utils functions for reading files."""

__author__ = ["TonyBagnall"]

__all__ = [
    "_alias_mtype_check",
    "_read_header",
    "_write_header",
    "write_results_to_uea_format",
]

import os
import pathlib
import textwrap
from typing import Union


def _alias_mtype_check(return_type):
    """Return appropriate return_type in case an alias was used."""
    if return_type is None:
        return_type = "nested_univ"
    if return_type in ["numpy2d", "numpy2D", "np2d", "np2D"]:
        return_type = "numpyflat"
    if return_type in ["numpy3d", "np3d", "np3D"]:
        return_type = "numpy3D"
    return return_type


# Do we need this function? I dont see it being used anywhere. research function?
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
                    raise OSError(
                        f"invalid timestamps tag value {tokens[1]} value in file "
                        f"{full_file_path_and_name}"
                    )
            elif line.startswith("@univariate"):
                tokens = line.split(" ")
                token_len = len(tokens)
                if tokens[1] == "false":
                    meta_data["is_univariate"] = False
                elif tokens[1] != "true":
                    raise OSError(
                        f"invalid univariate tag value {tokens[1]} in file "
                        f"{full_file_path_and_name}"
                    )
            elif line.startswith("@equallength"):
                tokens = line.split(" ")
                if tokens[1] == "false":
                    meta_data["is_equal_length"] = False
                elif tokens[1] != "true":
                    raise OSError(
                        f"invalid unequal tag value {tokens[1]} in file "
                        f"{full_file_path_and_name}"
                    )
            elif line.startswith("@classlabel"):
                tokens = line.split(" ")
                token_len = len(tokens)
                if tokens[1] == "false":
                    meta_data["has_class_labels"] = False
                elif tokens[1] != "true":
                    raise OSError(
                        f"invalid classLabel value in file {full_file_path_and_name}"
                    )
                if token_len == 2 and meta_data["class_labels"]:
                    raise OSError(
                        f"if the classlabel tag is true then class values must be "
                        f"supplied in file{full_file_path_and_name} but read {tokens}"
                    )
            elif line.startswith("@targetlabel"):
                tokens = line.split(" ")
                token_len = len(tokens)
                if tokens[1] == "false":
                    meta_data["has_class_labels"] = False
                elif tokens[1] != "true":
                    raise OSError(
                        f"invalid targetlabel value in file {full_file_path_and_name}"
                    )
                if token_len > 2:
                    raise OSError(
                        "targetlabel tag should not be accompanied with info "
                        "apart from true/false, but found "
                        f"{tokens}"
                    )
            elif line.startswith("@data"):
                return meta_data
    raise OSError(
        f"End of file reached for {full_file_path_and_name} but no indicated start of "
        f"data with the tag @data"
    )


def _write_header(
    path,
    problem_name,
    univariate,
    equal_length,
    series_length,
    class_label,
    fold,
    comment,
):
    """Write the header information for a ts file."""
    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}/"
    try:
        os.makedirs(dirt)
    except OSError:
        pass  # raises os.error if path already exists
    # create ts file in the path
    file = open(f"{dirt}{str(problem_name)}{fold}.ts", "w")
    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")

    """ Writes the header info for a ts file"""
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
        file.write("@classLabel false\n")
    file.write("@data\n")
    return file


# research function?
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
        determine file structure of output_root is True
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
        output_path = f"{output_path}/{estimator_name}/Predictions/{dataset_name}/"
    try:
        os.makedirs(output_path)
    except OSError:
        pass  # raises os.error if path already exists, so just ignore this

    if split == "TRAIN" or split == "train":
        train_or_test = "train"
    elif split == "TEST" or split == "test":
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")
    file = open(f"{output_path}/{train_or_test}Resample{resample_seed}.csv", "w")
    # the first line of the output file is in the form of:
    # <classifierName>,<datasetName>,<train/test>
    first_line = f"{dataset_name},{estimator_name},{train_or_test},{resample_seed}"
    if timing_type is not None:
        first_line += "," + timing_type
    if first_line_comment is not None:
        first_line += "," + first_line_comment
    file.write(first_line + "\n")
    # the second line of the output is free form and estimator-specific; usually this
    # will record info such as build time, parameter options used, any constituent model
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


def get_path(path: Union[str, pathlib.Path], suffix: str) -> str:
    """Automatic inference of file ending in data loaders for single file types.

    This function checks if the provided path has a specified suffix. If not,
    it checks if a file with the same name exists. If not, it adds the specified
    suffix to the path.

    Parameters
    ----------
    path: str or pathlib.Path
        The full pathname or filename.
    suffix: str
        The expected file extension.

    Returns
    -------
    resolved_path: str
        The filename with required extension
    """
    p_ = pathlib.Path(path).expanduser().resolve()
    resolved_path = str(p_)

    # Checks if the path has any extension
    if not p_.suffix:
        # Checks if a file with the same name exists
        if not os.path.exists(resolved_path):
            # adds the specified extension to the path
            resolved_path += suffix
    return resolved_path
