# -*- coding: utf-8 -*-
"""Utility functions for data handling."""

__author__ = "James Large"

import pandas as pd

from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.utils.validation.panel import check_X, check_X_y


def check_and_clean_data(X, y=None, input_checks=True):
    """Perform basic sktime data checks.

    Prepare the train data for input to keras models.

    Parameters
    ----------
    X: the train data
    y: the train labels
    input_checks: whether to perform the basic sktime checks

    Returns
    -------
    X
    """
    if input_checks:
        if y is None:
            check_X(X)
        else:
            check_X_y(X, y)

    # want data in form: [instances = n][timepoints = m][dimensions = d]
    if isinstance(X, pd.DataFrame):
        if _is_nested_dataframe(X):
            if X.shape[1] > 1:
                # we have multiple columns, AND each cell contains a series,
                # so this is a multidimensional problem
                X = _multivariate_nested_df_to_array(X)
            else:
                # we have a single column containing a series, treat this as
                # a univariate problem
                X = _univariate_nested_df_to_array(X)
        else:
            # we have multiple columns each containing a primitive, treat as
            # univariate series
            X = _univariate_df_to_array(X)

    if len(X.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        X = X.values.reshape(X.shape[0], X.shape[1], 1)  # go from [n][m] to [n][m][d=1]
    # return transposed data to conform with current model formats
    return X.transpose(0, 2, 1)


def check_and_clean_validation_data(
    validation_X,
    validation_y,
    label_encoder=None,
    onehot_encoder=None,
    input_checks=True,
):
    """Perform basic sktime data checks.

    Prepare the validation data for input to keras models.
    Also provide functionality to encode the y labels
    using label encoders that should have already been fit to the train data.

    :param validation_X: required, validation data
    :param validation_y: optional, y labels for categorical conversion if
            needed
    :param label_encoder: if validation_y has been given,
            the encoder that has already been fit to the train data
    :param onehot_encoder: if validation_y has been given,
            the encoder that has already been fit to the train data
    :param input_checks: whether to perform the basic input structure checks
    :return: ( validation_X, validation_y ), or None if no data given
    """
    if validation_X is not None:
        validation_X = check_and_clean_data(
            validation_X, validation_y, input_checks=input_checks
        )
    else:
        return None

    if label_encoder is not None and onehot_encoder is not None:
        validation_y = label_encoder.transform(validation_y)
        validation_y = validation_y.reshape(len(validation_y), 1)
        validation_y = onehot_encoder.transform(validation_y)

    return (validation_X, validation_y)


def _is_nested_dataframe(X):
    return isinstance(X.iloc[0, 0], pd.Series)


def _univariate_nested_df_to_array(X):
    return from_nested_to_3d_numpy(X)


def _univariate_df_to_array(X):
    return X.to_numpy()


def _multivariate_nested_df_to_array(X):
    return from_nested_to_3d_numpy(X)
