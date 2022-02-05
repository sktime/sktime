# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["Daniel Bartling"]

import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.datatypes import get_examples
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.window_summarizer import LaggedWindowSummarizer


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


# Load data that will be the basis of tests
y = load_airline()
y_pd = get_examples(mtype="pd.DataFrame", as_scitype="Series")[0]
y_series = get_examples(mtype="pd.Series", as_scitype="Series")[0]
y_multi = get_examples(mtype="pd-multiindex", as_scitype="Panel")[0]
# y Train will be univariate data set
y_train, y_test = temporal_train_test_split(y)

# Create Panel sample data
mi = pd.MultiIndex.from_product([[0], y.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y.values, index=mi, columns=["y"])

y_grouped = pd.concat([y_group1, y_group2])

y_ll, X_ll = load_longley()
y_ll_train, _, X_ll_train, X_ll_test = temporal_train_test_split(y_ll, X_ll)

# Get different WindowSummarizer functions
kwargs = LaggedWindowSummarizer.get_test_params()[0]
kwargs_alternames = LaggedWindowSummarizer.get_test_params()[1]
kwargs_variant = LaggedWindowSummarizer.get_test_params()[2]

# Generate named and unnamed y
y_train.name = None
y_train_named = y_train.copy()
y_train_named.name = "y"

# Target for multivariate extraction
Xt_test = ["POP_lag_3_0", "POP_lag_6_0", "GNP_lag_3_0", "GNP_lag_6_0"]
Xt_test = Xt_test + ["GNPDEFL", "UNEMP", "ARMED"]

# Some tests are commented out until hierarchical PR works


@pytest.mark.parametrize(
    "kwargs, column_names, y, target_cols",
    [
        (
            kwargs,
            ["y_lag_1_0", "y_mean_3_0", "y_mean_12_0", "y_std_4_0"],
            y_train_named,
            None,
        ),
        (kwargs_alternames, Xt_test, X_ll_train, ["POP", "GNP"]),
        # (kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_group1),
        # (kwargs, ["lag_1_0", "mean_3_0", "mean_12_0", "std_4_0"], y_grouped),
        # (None, ["lag_1_0"], y_multi),
        (None, None, y_train, None),
        (None, ["a_lag_1_0"], y_pd, None),
        (kwargs_alternames, ["var_0_lag_3_0", "var_0_lag_6_0"], y_train, None),
        (
            kwargs_variant,
            ["var_0_mean_7_0", "var_0_mean_7_7", "var_0_covar_feature_28_0"],
            y_train,
            None,
        ),
    ],
)
def test_windowsummarizer(kwargs, column_names, y, target_cols):
    """Test columns match kwargs arguments."""
    if kwargs is not None:
        transformer = LaggedWindowSummarizer(**kwargs, target_cols=target_cols)
    else:
        transformer = LaggedWindowSummarizer(target_cols=target_cols)
    Xt = transformer.fit_transform(y)
    if Xt is not None:
        if isinstance(Xt, pd.DataFrame):
            Xt_columns = Xt.columns.to_list()
        else:
            Xt_columns = Xt.name
    else:
        Xt_columns = None

    check_eval(Xt_columns, column_names)
