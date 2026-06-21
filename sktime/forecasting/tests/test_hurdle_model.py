# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for HurdleForecaster."""

import numpy as np
import pytest

from sktime.datasets import load_PBS_dataset
from sktime.forecasting.hurdle_model import HurdleForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HurdleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestHurdleForecaster:
    """Tests for HurdleForecaster."""

    @pytest.fixture
    def y(self):
        return load_PBS_dataset()

    def test_fit_predict_poisson(self, y):
        f = HurdleForecaster(alpha=0.2, beta=0.1, distribution="poisson")
        f.fit(y)
        y_pred = f.predict(fh=[1, 2, 3])
        assert len(y_pred) == 3
        assert (y_pred >= 0).all()

    def test_fit_predict_negbinom(self, y):
        f = HurdleForecaster(alpha=0.2, beta=0.1, distribution="negbinom")
        f.fit(y)
        y_pred = f.predict(fh=[1, 2, 3])
        assert len(y_pred) == 3
        assert (y_pred >= 0).all()

    def test_point_forecast_is_p_times_mu(self, y):
        f = HurdleForecaster(alpha=0.2, beta=0.1)
        f.fit(y)
        y_pred = f.predict(fh=[1])
        expected = f.demand_prob_ * f.demand_mean_
        np.testing.assert_almost_equal(float(y_pred.iloc[0]), expected, decimal=10)

    def test_constant_forecast(self, y):
        f = HurdleForecaster()
        f.fit(y)
        y_pred = f.predict(fh=[1, 2, 5, 10])
        assert np.allclose(y_pred.values, y_pred.values[0])

    def test_fitted_attributes(self, y):
        f = HurdleForecaster(alpha=0.3, beta=0.2)
        f.fit(y)
        assert 0.0 <= f.demand_prob_ <= 1.0
        assert f.demand_mean_ > 0.0

    def test_predict_interval_poisson(self, y):
        f = HurdleForecaster(alpha=0.2, beta=0.1, distribution="poisson")
        f.fit(y)
        pred_int = f.predict_interval(fh=[1, 2, 3], coverage=0.9)
        assert pred_int.shape[0] == 3
        var_name = f._get_varnames()[0]
        lower = pred_int[(var_name, 0.9, "lower")]
        upper = pred_int[(var_name, 0.9, "upper")]
        assert (lower <= upper).all()
        assert (lower >= 0).all()

    def test_predict_interval_negbinom(self, y):
        f = HurdleForecaster(alpha=0.2, beta=0.1, distribution="negbinom")
        f.fit(y)
        pred_int = f.predict_interval(fh=[1, 2, 3], coverage=0.9)
        var_name = f._get_varnames()[0]
        lower = pred_int[(var_name, 0.9, "lower")]
        upper = pred_int[(var_name, 0.9, "upper")]
        assert (lower <= upper).all()
        assert (lower >= 0).all()

    def test_predict_interval_multiple_coverages(self, y):
        f = HurdleForecaster()
        f.fit(y)
        pred_int = f.predict_interval(fh=[1], coverage=[0.5, 0.9])
        var_name = f._get_varnames()[0]
        lower_50 = float(pred_int[(var_name, 0.5, "lower")].iloc[0])
        upper_50 = float(pred_int[(var_name, 0.5, "upper")].iloc[0])
        lower_90 = float(pred_int[(var_name, 0.9, "lower")].iloc[0])
        upper_90 = float(pred_int[(var_name, 0.9, "upper")].iloc[0])
        assert lower_90 <= lower_50
        assert upper_90 >= upper_50

    def test_invalid_distribution_raises(self, y):
        f = HurdleForecaster(distribution="gamma")
        with pytest.raises(ValueError, match="distribution must be"):
            f.fit(y)

    @pytest.mark.parametrize("alpha,beta", [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)])
    def test_various_smoothing_params(self, y, alpha, beta):
        f = HurdleForecaster(alpha=alpha, beta=beta)
        f.fit(y)
        y_pred = f.predict(fh=[1])
        assert np.isfinite(float(y_pred.iloc[0]))
        assert float(y_pred.iloc[0]) >= 0

    def test_all_zeros_series(self):
        import pandas as pd

        y = pd.Series(np.zeros(20), dtype=float)
        f = HurdleForecaster()
        f.fit(y)
        y_pred = f.predict(fh=[1])
        assert float(y_pred.iloc[0]) >= 0

    def test_sparse_series(self):
        import pandas as pd

        rng = np.random.default_rng(42)
        vals = np.zeros(50)
        vals[[5, 15, 30, 45]] = rng.integers(1, 10, size=4).astype(float)
        y = pd.Series(vals)
        f = HurdleForecaster(alpha=0.1, beta=0.1)
        f.fit(y)
        y_pred = f.predict(fh=[1, 2, 3])
        assert (y_pred >= 0).all()
