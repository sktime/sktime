# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Testing of crafting functionality."""

__author__ = ["fkiraly"]

import pytest

from sktime.registry._craft import craft


pipe_spec = """
pipe = TransformedTargetForecaster(steps=[
    ("imputer", Imputer()),
    ("forecaster", NaiveForecaster())])
cv = ExpandingWindowSplitter(
    initial_window=24,
    step_length=12,
    fh=[1, 2, 3])
gscv = ForecastingGridSearchCV(
    forecaster=pipe,
    param_grid=[{
        "forecaster": [NaiveForecaster(sp=12)],
        "forecaster__strategy": ["drift", "last", "mean"],
    },
    {
        "imputer__method": ["mean", "drift"],
        "forecaster": [ThetaForecaster(sp=12)],
    },
    {
        "imputer__method": ["mean", "median"],
        "forecaster": [ExponentialSmoothing(sp=12)],
        "forecaster__trend": ["add", "mul"],
    },
    ],
    cv=cv,
    n_jobs=-1)
"""

specs = ["VAR(trend='ct')", pipe_spec]


@pytest.mark.parametrize("spec", specs)
def test_craft(spec):
    """Check that crafting works and is inverse to str coercion."""
    crafted_obj = craft(spec)

    new_spec = str(crafted_obj)

    crafted_again = craft(new_spec)

    assert crafted_again == crafted_obj
