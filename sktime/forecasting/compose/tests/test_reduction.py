# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["danbartl"]

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_airline
from sktime.datatypes import get_examples
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.summarize import WindowSummarizer

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

# Get different WindowSummarizer functions
kwargs = WindowSummarizer.get_test_params()[0]
kwargs_alternames = WindowSummarizer.get_test_params()[1]
kwargs_variant = WindowSummarizer.get_test_params()[2]

regressor = make_pipeline(
    RandomForestRegressor(),
)

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs)],
    window_length=10,
)

forecaster.fit(y_grouped, fh=1)

y_pred = forecaster.predict(fh=1)
