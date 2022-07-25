# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["danbartl"]

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_airline
from sktime.datatypes import get_examples
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.summarize import WindowSummarizer

# from sktime.utils.estimator_checks import check_estimator

# Load data that will be the basis of tests
y = load_airline()
y_multi = get_examples(mtype="pd-multiindex", as_scitype="Panel")[0]
# y Train will be univariate data set
y_train, y_test = temporal_train_test_split(y)


y_int = y.copy()
y_int.index = [i for i in range(len(y_int))]

y_train_int, y_test_int = temporal_train_test_split(y_int)

# Create Panel sample data
mi = pd.MultiIndex.from_product([[0], y_train.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y_train.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y_train.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y_train.values, index=mi, columns=["y"])

y_train_grp = pd.concat([y_group1, y_group2])

# Create Train Panel sample data
mi = pd.MultiIndex.from_product([[0], y_test.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y_test.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y_test.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y_test.values, index=mi, columns=["y"])

y_test_grp = pd.concat([y_group1, y_group2])

# Get different WindowSummarizer functions
kwargs = WindowSummarizer.get_test_params()[0]
kwargs_alternames = WindowSummarizer.get_test_params()[1]
kwargs_variant = WindowSummarizer.get_test_params()[2]

regressor = make_pipeline(
    RandomForestRegressor(random_state=1),
)

forecaster1 = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs)],
    window_length=None,
    strategy="recursive",
)

# forecaster1.fit(y=y_train_grp, X=y_train_grp)
# check_estimator(forecaster1, return_exceptions=False)
# y_pred1 = forecaster1.predict(X=y_test_grp, fh=[1, 2, 12])

forecaster2 = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs, n_jobs=1)],
    window_length=None,
    strategy="recursive",
)

y_numeric = y_train.copy()
y_numeric.index = pd.to_numeric(y_numeric.index)

# forecaster2.fit(y_train, fh=[1, 2])
# y_pred2 = forecaster2.predict(fh=[1, 2, 12])


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


@pytest.mark.parametrize(
    "y, index_names",
    [
        (
            y_train_grp,
            ["instances", "timepoints"],
        ),
        (
            y_train,
            [None],
        ),
        (
            y_numeric,
            [None],
        ),
    ],
)
def test_reduction(y, index_names):
    """Test index columns match input."""
    forecaster2.fit(y, fh=[1, 2])
    y_pred = forecaster2.predict(fh=[1, 2, 12])
    check_eval(y_pred.index.names, index_names)
