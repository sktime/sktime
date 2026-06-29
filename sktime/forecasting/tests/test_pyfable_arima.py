import pandas as pd

from sktime.forecasting.pyfable_arima import PyFableARIMA


def test_pyfablearima_formula_immutability(monkeypatch):
    # small monthly series
    idx = pd.period_range("2020-01", periods=6, freq="M")
    y = pd.Series([1, 2, 3, 4, 5, 6], index=idx, name="series")

    # remove hard dependency check for rpy2 for this isolated unit test
    monkeypatch.setitem(PyFableARIMA._tags, "python_dependencies", [])

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
