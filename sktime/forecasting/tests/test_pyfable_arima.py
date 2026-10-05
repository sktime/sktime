import sys
from types import ModuleType

import pandas as pd

import sktime.forecasting.base._base as forecasting_base
from sktime.forecasting.pyfable_arima import PyFableARIMA


def test_pyfablearima_formula_immutability(monkeypatch):
    # small monthly series
    idx = pd.period_range("2020-01", periods=6, freq="M")
    y = pd.Series([1, 2, 3, 4, 5, 6], index=idx, name="series")

    # bypass the hard rpy2 dependency check for this isolated unit test
    monkeypatch.setattr(
        forecasting_base, "_check_estimator_deps", lambda *args, **kwargs: True
    )

    # monkeypatch R interaction methods to avoid requiring actual R runtime here
    def dummy_prepare(self, Z, is_regular=True):
        return Z, Z  # placeholder

    def dummy_fit(self, train, expr):
        # store expr for inspection
        self._dummy_expr = expr
        return {"expr": expr}

    monkeypatch.setattr(PyFableARIMA, "_custom_prepare_tsibble", dummy_prepare)
    monkeypatch.setattr(PyFableARIMA, "_custom_fit_arima", dummy_fit)

    # Case 1: formula None, should remain None after fit, resolved stored
    f1 = PyFableARIMA(formula=None)
    assert f1.formula is None
    f1.fit(y, fh=[1])
    assert f1.formula is None, "formula attribute should remain None (immutable)"
    assert f1._resolved_formula == y.name, (
        "resolved formula should default to target name"
    )

    # Case 2: user-specified formula preserved
    f2 = PyFableARIMA(formula="series ~ 1")
    f2.fit(y, fh=[1])
    assert f2.formula == "series ~ 1"
    assert f2._resolved_formula == "series ~ 1"


def test_pyfablearima_report_evaluates_r_namespace_expression(monkeypatch):
    """Report uses an R expression, not a global-environment symbol lookup."""
    rpy2_module = ModuleType("rpy2")
    robjects_module = ModuleType("rpy2.robjects")
    r_calls = []

    def fake_r(expression):
        r_calls.append(expression)
        return ["ARIMA report", "Model: ARIMA(1,0,0)"]

    robjects_module.r = fake_r
    robjects_module.globalenv = {}
    rpy2_module.robjects = robjects_module
    monkeypatch.setitem(sys.modules, "rpy2", rpy2_module)
    monkeypatch.setitem(sys.modules, "rpy2.robjects", robjects_module)

    estimator = PyFableARIMA()
    fitted_model = object()
    estimator._fit_auto_arima_ = fitted_model

    report = estimator.report()

    assert robjects_module.globalenv["fit_aut_arima"] is fitted_model
    assert r_calls == ["capture.output(fabletools::report(fit_aut_arima))"]
    assert report == "ARIMA report\nModel: ARIMA(1,0,0)"
