# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Testing of crafting functionality."""

__author__ = ["fkiraly"]

import pytest

from sktime.registry._craft import craft, deps, imports
from sktime.utils.validation._dependencies import _check_soft_dependencies

simple_spec = "NaiveForecaster()"
simple_spec_with_dep = "VAR(trend='ct')"

pipe_spec = """
pipe = TransformedTargetForecaster(steps=[
    ("imputer", Imputer()),
    ("forecaster", NaiveForecaster())])
cv = ExpandingWindowSplitter(
    initial_window=24,
    step_length=12,
    fh=[1, 2, 3])

return ForecastingGridSearchCV(
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

dunder_spec = "Detrender(ExponentialSmoothing(sp=12)) * ARIMA()"

specs = [simple_spec, simple_spec_with_dep, pipe_spec, dunder_spec]


@pytest.mark.skipif(
    not _check_soft_dependencies(["statsmodels", "pmdarima"], severity="none"),
    reason="skip test if required soft dependencies not available",
)
@pytest.mark.parametrize("spec", specs)
def test_craft(spec):
    """Check that crafting works and is inverse to str coercion."""
    crafted_obj = craft(spec)

    new_spec = str(crafted_obj)

    crafted_again = craft(new_spec)

    assert crafted_again == crafted_obj


@pytest.mark.parametrize("spec", specs)
def test_deps(spec):
    """Check that deps retrieves the correct requirement sets."""
    # should return length 0 list since has no deps
    assert deps(simple_spec) == []

    # should correctly find the single dependency
    assert deps(simple_spec_with_dep) == ["statsmodels"]

    # has multiple estimators with "statsmodels",
    # this should be returned like this and not as ["statsmodels", "statsmodels"]
    assert deps(pipe_spec) == ["statsmodels"]

    # example with two dependencies, should be identified, order does not matter
    assert set(deps(dunder_spec)) == {"statsmodels", "pmdarima"}


def test_imports():
    """Check that imports produces the correct import blocks."""
    simple_spec_imports = "from sktime.forecasting.naive import NaiveForecaster"
    assert imports(simple_spec) == simple_spec_imports

    pipe_imports = (
        "from sktime.forecasting.compose._pipeline import TransformedTargetForecast"
        "er\nfrom sktime.forecasting.exp_smoothing import ExponentialSmoothing\nfrom"
        " sktime.forecasting.model_selection._tune import ForecastingGridSearch"
        "CV\nfrom sktime.forecasting.naive import NaiveForecaster\nfrom sktime.fore"
        "casting.naive import NaiveForecaster\nfrom sktime.forecasting.theta impor"
        "t ThetaForecaster\nfrom sktime.split.expandingwindow import "
        "ExpandingWindowSplitter\nfrom sktime.transformations.series.impute import "
        "Imputer"
    )
    assert imports(pipe_spec) == pipe_imports
