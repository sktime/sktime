"""Tests for sklearn init specification conformance in RBFForecaster.

Regression tests for #10208: parameters must be stored exactly as passed
to ``__init__``, so that ``get_params``, ``clone`` and ``set_params`` work.
"""

__author__ = ["Rythamo8055"]

from sklearn.base import clone

from sktime.forecasting.rbf import RBFForecaster


def test_rbf_forecaster_init_spec():
    """RBFForecaster stores hidden_layers as passed."""
    f = RBFForecaster()
    params = f.get_params()
    assert params["hidden_layers"] is None

    # clone round-trip preserves parameters
    cloned = clone(f)
    assert cloned.get_params() == params

    # explicit values are stored as-is
    f2 = RBFForecaster(hidden_layers=[16, 8])
    params2 = f2.get_params()
    assert params2["hidden_layers"] == [16, 8]
    assert clone(f2).get_params() == params2
