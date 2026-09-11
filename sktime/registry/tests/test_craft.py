# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Testing of crafting functionality."""

__author__ = ["fkiraly"]

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.registry._craft import craft, deps, imports

simple_spec = "NaiveForecaster()"
simple_spec_with_dep = "VAR(trend='ct')"

pipe_spec_no_deps = """
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
        "forecaster": [NaiveForecaster(sp=12)],
    },
    ],
    cv=cv,
    )
"""

pipe_spec_with_deps = """
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
    )
"""

dunder_spec_no_deps = "Imputer() * NaiveForecaster()"
dunder_spec_with_deps = (
    "Detrender(ExponentialSmoothing(sp=12)) * "
    "LTSFLinearForecaster(seq_len=10, pred_len=3)"
)

specs = [simple_spec, pipe_spec_no_deps, dunder_spec_no_deps]


if _check_soft_dependencies(["statsmodels"], severity="none"):
    specs += [simple_spec_with_dep, pipe_spec_with_deps, dunder_spec_with_deps]


@pytest.mark.parametrize("spec", specs)
@pytest.mark.parametrize("safe", [True, False])
def test_craft(spec, safe):
    """Check that crafting works and is inverse to str coercion."""
    # hack - among test cases, all unsafe specs contain a "return" statement
    # in general, this statement is not true, i.e., unsafe iff contains return
    spec_is_unsafe = "return" in spec

    # test that unsafe specs correctly raise an error in safe mode
    if safe and spec_is_unsafe:
        with pytest.raises(ValueError):
            craft(spec, safe=safe)
        return

    # test that crafting and re-crafting produces consistent results
    crafted_obj = craft(spec, safe=safe)

    new_spec = str(crafted_obj)

    crafted_again = craft(new_spec, safe=safe)

    assert crafted_again == crafted_obj


@pytest.mark.parametrize(
    "spec",
    [
        # Attribute access
        "NaiveForecaster().fit",
        "NaiveForecaster().foo()",
        "NaiveForecaster.__init__",

        # Indirect / arbitrary function calls
        "getattr(NaiveForecaster(), 'fit')",
        "(NaiveForecaster)()",
        "(lambda: NaiveForecaster())()",

        # Lambdas
        "lambda: NaiveForecaster()",
        "NaiveForecaster(lam=lambda: 1)",

        # Comprehensions / generators
        "[NaiveForecaster() for _ in range(1)]",
        "{NaiveForecaster() for _ in range(1)}",
        "{i: NaiveForecaster() for i in range(1)}",
        "(NaiveForecaster() for _ in range(1))",

        # Arbitrary builtin/function names are not in the safe registry
        "eval('NaiveForecaster()')",
        "exec('x = 1')",
        "open('foo')",
        "getattr",

        # Dunder names / access
        "__import__('os')",
        "__builtins__",
        "NaiveForecaster().__class__",
        "NaiveForecaster().__dict__",

        # **kwargs expansion
        "NaiveForecaster(**{})",

        # Statements / multi-statement code
        "x = NaiveForecaster()",
        "NaiveForecaster(); NaiveForecaster()",
        "import os",
        "from os import path",
        "if True:\n    return NaiveForecaster()",

        # Unsupported expression forms
        "[NaiveForecaster()]",
        "{'estimator': NaiveForecaster()}",
        "(NaiveForecaster(),)",
        "NaiveForecaster() if True else NaiveForecaster()",

        # Boolean operators are deliberately not part of the safe grammar
        "NaiveForecaster() and NaiveForecaster()",
        "NaiveForecaster() or NaiveForecaster()",

        # Comparisons
        "NaiveForecaster() == NaiveForecaster()",
        "NaiveForecaster() < NaiveForecaster()",

        # Chained / indirect calls
        "NaiveForecaster()()",
        "(NaiveForecaster())()",
    ],
)
def test_craft_safe_rejects_unsafe_specs(spec):
    """Test that unsafe Python constructs are rejected in safe mode."""
    with pytest.raises(ValueError, match="unsafe or invalid specification|safe mode"):
        craft(spec, safe=True)


@pytest.mark.parametrize("spec", specs)
def test_deps(spec):
    """Check that deps retrieves the correct requirement sets."""
    # should return length 0 list since has no deps
    assert deps(simple_spec) == []
    assert deps(pipe_spec_no_deps) == []
    assert deps(dunder_spec_no_deps) == []

    # should correctly find the single dependency
    assert deps(simple_spec_with_dep) == ["statsmodels"]

    # has multiple estimators with "statsmodels",
    # this should be returned like this and not as ["statsmodels", "statsmodels"]
    assert deps(pipe_spec_with_deps) == ["statsmodels"]

    # example with two dependencies, should be identified, order does not matter
    expected_deps = {"statsmodels", "torch"}
    assert set(deps(dunder_spec_with_deps)) == expected_deps


def test_imports():
    """Check that imports produces the correct import blocks."""
    simple_spec_imports = "from sktime.forecasting.naive import NaiveForecaster"
    assert imports(simple_spec) == simple_spec_imports

    pipe_imports = (
        "from sktime.forecasting.compose import TransformedTargetForecast"
        "er\nfrom sktime.forecasting.exp_smoothing import ExponentialSmoothing\nfrom"
        " sktime.forecasting.model_selection import ForecastingGridSearch"
        "CV\nfrom sktime.forecasting.naive import NaiveForecaster\nfrom sktime.fore"
        "casting.naive import NaiveForecaster\nfrom sktime.forecasting.theta impor"
        "t ThetaForecaster\nfrom sktime.split.expandingwindow import "
        "ExpandingWindowSplitter\nfrom sktime.transformations.impute import "
        "Imputer"
    )
    assert imports(pipe_spec_with_deps) == pipe_imports


def test_deps_with_disjunction():
    """Check that deps retrieves the correct requirement set for disjunctions."""
    assert set(deps("DartsXGBModel")) == {"xgboost", "u8darts>=0.29"}


def test_sklearn_imports():
    """Check that sklearn estimators can be crafted."""
    from sktime.registry._lookup_sklearn import _all_sklearn_estimators

    sklearn_estimators = dict(_all_sklearn_estimators())

    from sklearn.ensemble import RandomForestRegressor

    assert craft("RandomForestRegressor()").__class__ == RandomForestRegressor
    rf_instance = craft("RandomForestRegressor(n_estimators=10)")
    assert isinstance(rf_instance, RandomForestRegressor)
    assert craft("RandomForestRegressor(n_estimators=10)").n_estimators == 10

    for est_name in ["StandardScaler", "KNeighborsClassifier", "RandomForestRegressor"]:
        assert est_name in sklearn_estimators.keys()

        est_spec = f"{est_name}()"
        est_obj = craft(est_spec)

        assert est_obj.__class__.__name__ == est_name
