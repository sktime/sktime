# -*- coding: utf-8 -*-
import itertools
import os
import textwrap

import pandas as pd
from sklearn.metrics import accuracy_score as acc

__author__ = ["Jason Pong"]


def write_results_to_uea_format(
    path,
    strategy_name,
    dataset_name,
    y_true,
    y_pred,
    split="TEST",
    resample_seed=0,
    y_proba=None,
    second_line="N/A",
):
    if len(y_true) != len(y_pred):
        raise IndexError(
            "The number of predicted class values is not the same as the "
            "number of actual class values"
        )

    try:
        os.makedirs(
            str(path)
            + "/"
            + str(strategy_name)
            + "/Predictions/"
            + str(dataset_name)
            + "/"
        )
    except os.error:
        pass  # raises os.error if path already exists

    if split == "TRAIN" or split == "train":
        train_or_test = "train"
    elif split == "TEST" or split == "test":
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")

    file = open(
        str(path)
        + "/"
        + str(strategy_name)
        + "/Predictions/"
        + str(dataset_name)
        + "/"
        + str(train_or_test)
        + "Fold"
        + str(resample_seed)
        + ".csv",
        "w",
    )

    correct = acc(y_true, y_pred)

    # the first line of the output file is in the form of:
    # <classifierName>,<datasetName>,<train/test>
    file.write(
        str(strategy_name) + "," + str(dataset_name) + "," + str(train_or_test) + "\n"
    )

    # the second line of the output is free form and classifier-specific;
    # usually this will record info
    # such as build time, paramater options used, any constituent model
    # names for ensembles, etc.
    file.write(str(second_line) + "\n")

    # the third line of the file is the accuracy (should be between 0 and 1
    # inclusive). If this is a train
    # output file then it will be a training estimate of the classifier on
    # the training data only (e.g.
    # 10-fold cv, leave-one-out cv, etc.). If this is a test output file,
    # it should be the output
    # of the estimator on the test data (likely trained on the training data
    # for a-priori parameter optimisation)

    file.write(str(correct) + "\n")

    # from line 4 onwards each line should include the actual and predicted
    # class labels (comma-separated). If
    # present, for each case, the probabilities of predicting every class
    # value for this case should also be
    # appended to the line (a space is also included between the predicted
    # value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   actual_class_val[i], predicted_class_val[i],,prob_class_0[i],
    #   prob_class_1[i],...,prob_class_c[i]
    #
    # if predict_proba data IS NOT provided for case i:
    #   actual_class_val[i], predicted_class_val[i]
    for i in range(0, len(y_pred)):
        file.write(str(y_true[i]) + "," + str(y_pred[i]))
        if y_proba is not None:
            file.write(",")
            for j in y_proba[i]:
                file.write("," + str(j))
            file.write("\n")  # TODO BUG new line is written only if the
            # probas are provided!!!!

    file.close()


def write_dataframe_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    timestamp=False,
    univariate=True,
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
):
    """
    Output a dataset in dataframe format to .ts file
    Parameters
    ----------
    data: pandas dataframe
        the dataset in a dataframe to be written as a ts file
        which must be of the structure specified in the documentation
        https://github.com/whackteachers/sktime/blob/master/examples/loading_data.ipynb
        index |   dim_0   |   dim_1   |    ...    |  dim_c-1
           0  | pd.Series | pd.Series | pd.Series | pd.Series
           1  | pd.Series | pd.Series | pd.Series | pd.Series
          ... |    ...    |    ...    |    ...    |    ...
           n  | pd.Series | pd.Series | pd.Series | pd.Series
    path: str
        The full path to output the ts file
    problem_name: str
        The problemName to print in the header of the ts file
        and also the name of the file.
    timestamp: {False, bool}, optional
        Indicate whether the data contains timestamps in the header.
    univariate: {True, bool}, optional
        Indicate whether the data is univariate or multivariate in the header.
        If univariate, only the first dimension will be written to file
    class_label: {list, None}, optional
        Provide class label to show the possible class values
        for classification problems in the header.
    class_value_list: {list/ndarray, []}, optional
        ndarray containing the class values for each case in classification problems
    equal_length: {False, bool}, optional
        Indicate whether each series has equal length. It only write to file if true.
    series_length: {-1, int}, optional
        Indicate each series length if they are of equal length.
        It only write to file if true.
    missing_values: {NaN, str}, optional
        Representation for missing value, default is NaN.
    comment: {None, str}, optional
        Comment text to be inserted before the header in a block.

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.

    References
    ----------
    The code for writing series data into file is adopted from
    https://stackoverflow.com/questions/37877708/
    how-to-turn-a-pandas-dataframe-row-into-a-comma-separated-string
    """
    if class_value_list is None:
        class_value_list = []
    # ensure data provided is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data provided must be a DataFrame")
    # ensure number of cases is same as the class value list
    if len(data.index) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the " "number of given class values"
        )

    if equal_length and series_length == -1:
        raise ValueError(
            "Please specify the series length for equal length time series data."
        )

    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}/"
    try:
        os.makedirs(dirt)
    except os.error:
        pass  # raises os.error if path already exists

    # create ts file in the path
    file = open(f"{dirt}{str(problem_name)}_transform.ts", "w")

    # write comment if any as a block at start of file
    if comment:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")
    # begin writing header information
    file.write(f"@problemName {problem_name}\n")
    file.write(f"@timeStamps {str(timestamp).lower()}\n")
    file.write(f"@univariate {str(univariate).lower()}\n")

    # write equal length or series length if provided
    if equal_length:
        file.write(f"@equalLength {str(equal_length).lower()}\n")
    if series_length > 0:
        file.write(f"@seriesLength {series_length}\n")

    # write class label line
    if class_label:
        space_separated_class_label = " ".join(str(label) for label in class_label)
        file.write(f"@classLabel true {space_separated_class_label}\n")
    else:
        file.write("@class_label false\n")

    # begin writing the core data for each case
    # which are the series and the class value list if there is any
    file.write("@data\n")
    for case, value in itertools.zip_longest(data.iterrows(), class_value_list):
        for dimension in case[1:]:  # start from the first dimension
            # split the series observation into separate token
            # ignoring the header and index
            series = (
                dimension[0]
                .to_string(index=False, header=False, na_rep=missing_values)
                .split("\n")
            )
            # turn series into comma-separated row
            series = ",".join(obsv for obsv in series)
            file.write(str(series))
            # continue with another dimension for multivariate case
            if not univariate:
                file.write(":")
        if value is not None:
            file.write(f":{value}")  # write the case value if any
        file.write("\n")  # open a new line

    file.close()


if __name__ == "__main__":
    actual = [1, 1, 2, 2, 1, 1, 2, 2]
    preds = [1, 1, 2, 2, 1, 2, 1, 2]
    probas = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.7, 0.3],
        [0.2, 0.8],
    ]

    write_results_to_uea_format(
        path="../exampleResults",
        strategy_name="dummy_classifier",
        dataset_name="banana_point",
        y_true=actual,
        y_pred=preds,
        split="TEST",
        resample_seed=0,
        y_proba=probas,
        second_line="buildTime=100000,num_dummy_things=2",
    )
