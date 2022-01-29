# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["Daniel Bartling"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import get_examples
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.panel.window_summarizer import LaggedWindowSummarizer


def check_eval(test_input, expected):
    """Test which columns are returned for different arguments.

    For a detailed description what these arguments do,
    and how theyinteract see docstring of DateTimeFeatures.
    """
    if test_input is not None:
        assert len(test_input) == len(expected)
        assert all([a == b for a, b in zip(test_input, expected)])
    else:
        assert expected is None


# Load data that will be basis of tests
y = load_airline()
y_pd = get_examples(mtype="pd.DataFrame", as_scitype="Series")[0]
y_series = get_examples(mtype="pd.Series", as_scitype="Series")[0]
y_multi = get_examples(mtype="pd-multiindex", as_scitype="Panel")[0]
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

transformer = LaggedWindowSummarizer()
Xt = transformer.fit_transform(y_series)


@pytest.mark.parametrize(
    "kwargs, column_names, y",
    [
        (kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_train),
        #(kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_group1),
        #(kwargs, ["lag_1_0", "
        # mean_3_0", "mean_12_0", "std_4_0"], y_grouped),
        (None, ["lag_1_0"], y_pd),
        (None, ["lag_1_0"], y_series),
        # (None, ["lag_1_0"], y_multi),
        (kwargs_alternames, ["lag_3_0", "lag_12_0"], y_train),
        (kwargs_variant, ["mean_7_0", "mean_7_7", "covar_feature_28_0"], y_train),
    ],
)
def test_windowsummarizer(kwargs, column_names, y):
    """Test columns match kwargs arguments."""
    if kwargs is not None:
        transformer = LaggedWindowSummarizer(**kwargs)
    else:
        transformer = LaggedWindowSummarizer()
    Xt = transformer.fit_transform(y_train)
    if Xt is not None:
        Xt_columns = Xt.columns.to_list()
    else:
        Xt_columns = None

    check_eval(Xt_columns, column_names)
