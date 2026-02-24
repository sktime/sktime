# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TabICLForecaster.

Run with:
    pytest sktime/forecasting/tests/test_tabicl.py -v

Or against only this estimator via the standard sktime check:
    from sktime.utils.estimator_checks import check_estimator
    from sktime.forecasting.tabicl import TabICLForecaster
    check_estimator(TabICLForecaster, raise_exceptions=True)
"""
import pytest

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tabicl", severity="none"),
    reason="tabicl not installed",
)
class TestTabICLForecaster:
    """Tests for TabICLForecaster."""

    def test_fit_predict_basic(self):
        """Test basic fit/predict cycle with default parameters."""
        from sktime.datasets import load_airline
        from sktime.forecasting.tabicl import TabICLForecaster

        y = load_airline()
        forecaster = TabICLForecaster(window_length=3, n_estimators=1, use_kv_cache=False)
        forecaster.fit(y, fh=[1, 2, 3])
        y_pred = forecaster.predict()

        assert len(y_pred) == 3
        assert not y_pred.isna().any()

    def test_fit_predict_single_step(self):
        """Test single-step ahead forecast."""
        from sktime.datasets import load_airline
        from sktime.forecasting.tabicl import TabICLForecaster

        y = load_airline()
        forecaster = TabICLForecaster(window_length=4, n_estimators=1, use_kv_cache=False)
        forecaster.fit(y, fh=[1])
        y_pred = forecaster.predict()

        assert len(y_pred) == 1
        assert not y_pred.isna().any()

    def test_window_too_large_raises(self):
        """Raise ValueError when window_length >= len(y)."""
        import numpy as np
        import pandas as pd
        from sktime.forecasting.tabicl import TabICLForecaster

        y = pd.Series(np.random.randn(5))
        forecaster = TabICLForecaster(window_length=5, n_estimators=1)

        with pytest.raises(ValueError, match="window_length"):
            forecaster.fit(y, fh=[1])

    def test_fh_passed_late(self):
        """Forecasting horizon can be passed in predict, not fit."""
        from sktime.datasets import load_airline
        from sktime.forecasting.tabicl import TabICLForecaster

        y = load_airline()
        forecaster = TabICLForecaster(window_length=3, n_estimators=1, use_kv_cache=False)
        forecaster.fit(y)
        y_pred = forecaster.predict(fh=[1, 2])

        assert len(y_pred) == 2

    def test_non_contiguous_fh(self):
        """Test with non-contiguous forecasting horizon like [1, 3, 6]."""
        from sktime.datasets import load_airline
        from sktime.forecasting.tabicl import TabICLForecaster

        y = load_airline()
        forecaster = TabICLForecaster(window_length=4, n_estimators=1, use_kv_cache=False)
        forecaster.fit(y, fh=[1, 3, 6])
        y_pred = forecaster.predict()

        assert len(y_pred) == 3

    def test_make_tabular_shape(self):
        """Unit test the _make_tabular module-level function."""
        import numpy as np
        from sktime.forecasting.tabicl import _make_tabular

        values = np.arange(20, dtype=float)
        X_tab, y_tab = _make_tabular(values, window_length=5)

        assert X_tab.shape == (15, 5), f"Expected (15, 5), got {X_tab.shape}"
        assert y_tab.shape == (15,), f"Expected (15,), got {y_tab.shape}"
        assert list(X_tab[0]) == [0, 1, 2, 3, 4]
        assert y_tab[0] == 5.0

    def test_impute_array_no_nans(self):
        """_impute_array returns original array when no NaNs present."""
        import numpy as np
        from sktime.forecasting.tabicl import _impute_array

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _impute_array(X)
        np.testing.assert_array_equal(result, X)

    def test_impute_array_with_nans(self):
        """_impute_array replaces NaNs with column means."""
        import numpy as np
        from sktime.forecasting.tabicl import _impute_array

        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = _impute_array(X)
        assert not np.isnan(result).any()
        assert result[0, 1] == 4.0

    def test_get_test_params(self):
        """get_test_params returns a list of two dicts."""
        from sktime.forecasting.tabicl import TabICLForecaster

        params = TabICLForecaster.get_test_params()
        assert isinstance(params, list)
        assert len(params) == 2
        for p in params:
            assert isinstance(p, dict)
            assert "window_length" in p