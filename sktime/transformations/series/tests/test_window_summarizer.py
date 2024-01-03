"""Test extraction of features across (shifted) windows."""
__author__ = ["danbartl"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.datatypes import get_examples
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.summarize import WindowSummarizer


def check_eval(test_input, expected):
    """Test which columns are returned for different arguments.

    For a detailed description what these arguments do, and how theyinteract see
    docstring of DateTimeFeatures.
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

mi = pd.MultiIndex.from_product([[0], [0], y.index], names=["h1", "h2", "time"])
y_hier1 = pd.DataFrame(y.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[0], [1], y.index], names=["h1", "h2", "time"])
y_hier2 = pd.DataFrame(y.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], [0], y.index], names=["h1", "h2", "time"])
y_hier3 = pd.DataFrame(y.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], [1], y.index], names=["h1", "h2", "time"])
y_hier4 = pd.DataFrame(y.values, index=mi, columns=["y"])

y_hierarchical = pd.concat([y_hier1, y_hier2, y_hier3, y_hier4])

y_ll, X_ll = load_longley()
y_ll_train, _, X_ll_train, X_ll_test = temporal_train_test_split(y_ll, X_ll)

# Get different WindowSummarizer functions
kwargs = WindowSummarizer.get_test_params()[0]
kwargs_alternames = WindowSummarizer.get_test_params()[1]
kwargs_variant = WindowSummarizer.get_test_params()[2]


def count_gt100(x):
    """Count how many observations lie above threshold 100."""
    return np.sum((x > 100)[::-1])


# Cannot be pickled in get_test_params, therefore here explicit
kwargs_custom = {
    "lag_feature": {
        count_gt100: [[3, 2]],
    }
}
# Generate named and unnamed y
y_train.name = None
y_train_named = y_train.copy()
y_train_named.name = "y"

# Target for multivariate extraction
Xtmvar = ["POP_lag_3", "POP_lag_6", "GNP_lag_3", "GNP_lag_6"]
Xtmvar = Xtmvar + ["GNPDEFL", "UNEMP", "ARMED"]
Xtmvar_none = ["GNPDEFL_lag_3", "GNPDEFL_lag_6", "GNP", "UNEMP", "ARMED", "POP"]


@pytest.mark.skipif(
    not run_test_for_class(WindowSummarizer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "kwargs, column_names, y, target_cols, truncate",
    [
        (
            kwargs,
            ["y_lag_1", "y_mean_1_3", "y_mean_1_12", "y_std_1_4"],
            y_train_named,
            None,
            None,
        ),
        (kwargs_alternames, Xtmvar, X_ll_train, ["POP", "GNP"], None),
        (kwargs_alternames, Xtmvar_none, X_ll_train, None, None),
        (
            kwargs,
            ["y_lag_1", "y_mean_1_3", "y_mean_1_12", "y_std_1_4"],
            y_group1,
            None,
            None,
        ),
        (
            kwargs,
            ["y_lag_1", "y_mean_1_3", "y_mean_1_12", "y_std_1_4"],
            y_grouped,
            None,
            None,
        ),
        (
            kwargs,
            ["y_lag_1", "y_mean_1_3", "y_mean_1_12", "y_std_1_4"],
            y_hierarchical,
            None,
            None,
        ),
        (
            None,
            ["var_0_lag_1", "var_1"],
            y_multi,
            None,
            None,
        ),
        (None, None, y_train, None, None),
        (None, ["a_lag_1"], y_pd, None, None),
        (kwargs_custom, ["a_count_gt100_3_2"], y_pd, None, None),
        (kwargs_alternames, ["0_lag_3", "0_lag_6"], y_train, None, "bfill"),
        (
            kwargs_variant,
            ["0_mean_1_7", "0_mean_8_7", "0_cov_1_28"],
            y_train,
            None,
            None,
        ),
    ],
)
def test_windowsummarizer(kwargs, column_names, y, target_cols, truncate):
    """Test columns match kwargs arguments."""
    if kwargs is not None:
        transformer = WindowSummarizer(
            **kwargs, target_cols=target_cols, truncate=truncate
        )
    else:
        transformer = WindowSummarizer(target_cols=target_cols, truncate=truncate)
    Xt = transformer.fit_transform(y)
    if Xt is not None:
        if isinstance(Xt, pd.DataFrame):
            Xt_columns = Xt.columns.to_list()
        else:
            Xt_columns = Xt.name
    else:
        Xt_columns = None

    # check that the index names are preserved
    assert y.index.names == Xt.index.names

    check_eval(Xt_columns, column_names)


@pytest.mark.xfail(raises=ValueError)
def test_wrong_column():
    """Test mismatch between X column names and target_cols."""
    transformer = WindowSummarizer(target_cols=["dummy"])
    Xt = transformer.fit_transform(X_ll_train)
    return Xt
