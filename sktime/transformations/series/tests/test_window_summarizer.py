# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["Daniel Bartling"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.window_summarizer import LaggedWindowSummarizer

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

# Create sample function dictionaries
kwargs = {
    "functions": {
        "lag": ["lag", [[1, 1]]],
        "mean": ["mean", [[1, 1], [5, 1]]],
        "std": ["std", [[1, 2]]],
    }
}

kwargs_alternames = {
    "functions": {
        "lag": ["lag", [[1, 1], [2, 1], [4, 1]]],
        "covar": ["cov", [[1, 1], [2, 1], [4, 1]]],
    }
}

kwargs_empty = {"functions": {}}

# Create transformer with kwargs
transformer = LaggedWindowSummarizer(**kwargs)

# Create transformer with different set of kwargs
transformer_alternames = LaggedWindowSummarizer(**kwargs_alternames)

# Create empty transformer
transformer_empty = LaggedWindowSummarizer(**kwargs_empty)

# Check if transformers works with non-Panel date
Xt_univar = transformer.fit_transform(y_train)
test_univar = Xt_univar.columns.to_list()

# Check if transformers works with single instance Panel data
Xt_singlegroup = transformer.fit_transform(y_group1)
test_singlegroup = Xt_singlegroup.columns.to_list()

# Check if transformers works with multi instance Panel data
Xt_multigroup = transformer.fit_transform(y_grouped)
test_multigroup = Xt_multigroup.columns.to_list()

# Check if transformers works with Panel data and empty function dictionary
Xt_empty = transformer_empty.fit_transform(y_grouped)
test_empty = Xt_empty.columns.to_list()

# Check if transformers works with Panel data and alternate function dictionary
Xt_alternames = transformer_alternames.fit_transform(y_grouped)
test_alternames = Xt_alternames.columns.to_list()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            test_univar,
            ["lag_1_1", "mean_1_1", "mean_5_1", "std_1_2"],
        ),
        (
            test_singlegroup,
            ["lag_1_1", "mean_1_1", "mean_5_1", "std_1_2"],
        ),
        (
            test_multigroup,
            ["lag_1_1", "mean_1_1", "mean_5_1", "std_1_2"],
        ),
        (
            test_alternames,
            ["lag_1_1", "lag_2_1", "lag_4_1", "covar_1_1", "covar_2_1", "covar_4_1"],
        ),
        (test_empty, ["y"]),
    ],
)
def test_eval(test_input, expected):
    """Tests which columns are returned for different arguments.

    For a detailed description what these arguments do,
    and how theyinteract see docstring of DateTimeFeatures.
    """
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])
