# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for FlowStateForecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.flowstate import FlowStateForecaster


class _FlowStateWithoutModel(FlowStateForecaster):
    """FlowState test double that avoids loading the external model."""

    _tags = {**FlowStateForecaster._tags, "python_dependencies": []}

    def _fit(self, y, X=None, fh=None):
        self._context = y
        return self

    def _predict(self, fh, X=None):
        index = fh.to_absolute(self.cutoff)._values
        return pd.DataFrame(0.0, index=index, columns=self._context.columns)

    def _predict_quantiles(self, fh, X, alpha):
        index = fh.to_absolute(self.cutoff)._values
        cols = pd.MultiIndex.from_product([self._context.columns, alpha])
        return pd.DataFrame(np.tile(alpha, (len(index), 1)), index=index, columns=cols)


def test_flowstate_predict_proba_does_not_use_normal_fallback():
    """FlowState predict_proba should not expose the base Normal fallback."""
    y = pd.DataFrame({"y": np.arange(10.0)})
    forecaster = _FlowStateWithoutModel().fit(y, fh=[1, 2])

    with pytest.raises(NotImplementedError, match="HistogramQPD"):
        forecaster.predict_proba()


def test_flowstate_predict_var_still_uses_quantiles():
    """FlowState variance should remain available from quantile forecasts."""
    y = pd.DataFrame({"y": np.arange(10.0)})
    forecaster = _FlowStateWithoutModel().fit(y, fh=[1, 2])

    pred_var = forecaster.predict_var()

    assert list(pred_var.columns) == ["y"]
    assert pred_var.index.equals(pd.Index([10, 11]))
    assert (pred_var["y"] > 0).all()
