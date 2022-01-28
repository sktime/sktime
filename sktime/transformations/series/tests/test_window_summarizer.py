# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["Daniel Bartling"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.window_summarizer import LaggedWindowSummarizer


def check_eval(test_input, expected):
    """Test which columns are returned for different arguments.

    For a detailed description what these arguments do,
    and how theyinteract see docstring of DateTimeFeatures.
    """
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])


# Load data that will be basis of tests
y = load_airline()

# Y Train will be univariate data set
y_train, y_test = temporal_train_test_split(y)

# Create Panel sample data
mi = pd.MultiIndex.from_product([[0], y.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y.values, index=mi, columns=["y"])

y_grouped = pd.concat([y_group1, y_group2])

kwargs = {
    "functions": {
        "lag": ["lag", [[1, 0]]],
        "mean": ["mean", [[3, 0], [12, 0]]],
        "std": ["std", [[4, 0]]],
    }
}

kwargs_alternames = {
    "functions": {
        "lag": ["lag", [[3, 0], [12, 0]]],
    }
}

kwargs_variant = {
    "functions": {
        "mean": ["mean", [[7, 0], [7, 7]]],
        "covar_feature": ["cov", [[28, 0]]],
    }
}

kwargs_empty = {"functions": {}}


@pytest.mark.parametrize(
    "kwargs, column_names, y",
    [
        (kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_train),
        (kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_group1),
        (kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_grouped),
        # (kwargs_empty, ["y"], y_grouped),
        (kwargs_alternames, ["lag_3_0", "lag_12_0"], y_train),
        (kwargs_variant, ["mean_7_0", "mean_7_7", "covar_feature_28_0"], y_train),
    ],
)
def test_univariate(kwargs, column_names, y):
    """Test columns match kwargs arguments."""
    transformer = LaggedWindowSummarizer(**kwargs)

    Xt = transformer.fit_transform(y_train)
    Xt_columns = Xt.columns.to_list()

    check_eval(Xt_columns, column_names)
