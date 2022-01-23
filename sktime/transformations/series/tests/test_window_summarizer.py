# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["Daniel Bartling"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.window_summarizer import LaggedWindowSummarizer

kwargs = {
    "functions": {
        "lag": ["lag", [[1, 1]]],
        "mean": ["mean", [[1, 1], [5, 1]]],
        "std": ["std", [[1, 2]]],
    }
}

y = load_airline()
y_train, y_test = temporal_train_test_split(y)


transformer = LaggedWindowSummarizer(**kwargs)

mi = pd.MultiIndex.from_product([[0], y.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y.values, index=mi, columns=["y"])

y_grouped = pd.concat([y_group1, y_group2])

Xt_nongroup = transformer.fit_transform(y_group1)
# print(Xt)

Xt_group = transformer.fit_transform(y_grouped)

test_singlegroup = Xt_nongroup.columns.to_list()

test_multigroup = Xt_nongroup.columns.to_list()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            test_singlegroup,
            ["lag_1_1", "mean_1_1", "mean_5_1", "std_1_2"],
        ),
        (
            test_multigroup,
            ["lag_1_1", "mean_1_1", "mean_5_1", "std_1_2"],
        ),
    ],
)
def test_eval(test_input, expected):
    """Tests which columns are returned for different arguments.

    For a detailed description what these arguments do,
    and how theyinteract see docstring of DateTimeFeatures.
    """
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])
