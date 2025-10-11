#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License
"""Tests for DoubleMLForecaster."""

__author__ = ["XAheli"]

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sktime.datasets import load_longley
from sktime.forecasting.compose import DoubleMLForecaster, make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.estimator_checks import check_estimator


# @pytest.mark.skipif(
#     not run_test_for_class(DoubleMLForecaster),
#     reason="run test only if softdeps are present and incrementally (if requested)",
# )
class TestDoubleMLForecaster:
    """Test class for DoubleMLForecaster."""

    def test_dml_forecaster_basic(self):
        """Test basic functionality of DoubleMLForecaster."""
        # Load test data
        y, X = load_longley()

        # Create basic DoubleMLForecaster
        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            exposure_vars=["GNP"],
        )

        # Test fit and predict
        forecaster.fit(y, X=X, fh=[1, 2, 3])
        y_pred = forecaster.predict(fh=[1, 2, 3], X=X)

        assert len(y_pred) == 3
        assert y_pred.index.equals(y.index[-3:] + 1)

    def test_dml_forecaster_with_make_reduction(self):
        """Test DoubleMLForecaster with make_reduction forecasters."""
        y, X = load_longley()

        forecaster = DoubleMLForecaster(
            forecaster_y=make_reduction(
                RandomForestRegressor(n_estimators=5, random_state=42)
            ),
            forecaster_ex=make_reduction(LinearRegression()),
            exposure_vars=["GNP"],
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        y_pred = forecaster.predict(fh=[1, 2], X=X)

        assert len(y_pred) == 2

    def test_dml_forecaster_error_handling(self):
        """Test error handling in DoubleMLForecaster."""
        # Test missing exposure vars
        with pytest.raises(ValueError, match="exposure_vars must be specified"):
            DoubleMLForecaster(
                forecaster_y=NaiveForecaster(),
                forecaster_ex=NaiveForecaster(),
                exposure_vars=[],
            )

        # Test missing exposure vars in X
        y, X = load_longley()
        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(),
            forecaster_ex=NaiveForecaster(),
            exposure_vars=["NONEXISTENT"],
        )

        with pytest.raises(ValueError, match="not found in X columns"):
            forecaster.fit(y, X=X, fh=[1])

    def test_dml_forecaster_prediction_intervals(self):
        """Test prediction intervals functionality."""
        y, X = load_longley()

        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            exposure_vars=["GNP"],
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        pred_int = forecaster.predict_interval(fh=[1, 2], X=X, coverage=0.95)

        assert pred_int.shape == (2, 2)  # 2 time points, 2 columns (lower, upper)

    def test_dml_forecaster_prediction_quantiles(self):
        """Test prediction quantiles functionality."""
        y, X = load_longley()

        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            exposure_vars=["GNP"],
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        pred_quantiles = forecaster.predict_quantiles(
            fh=[1, 2], X=X, alpha=[0.1, 0.5, 0.9]
        )

        assert pred_quantiles.shape == (2, 3)  # 2 time points, 3 quantiles

    def test_dml_forecaster_prediction_variance(self):
        """Test prediction variance functionality."""
        y, X = load_longley()

        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            exposure_vars=["GNP"],
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        pred_var = forecaster.predict_var(fh=[1, 2], X=X)

        assert pred_var.shape == (2, 1)  # 2 time points, 1 variance column

    def test_dml_forecaster_multiple_exposure_vars(self):
        """Test DoubleMLForecaster with multiple exposure variables."""
        y, X = load_longley()

        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            exposure_vars=["GNP", "GNPDEFL"],  # Multiple exposure vars
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        y_pred = forecaster.predict(fh=[1, 2], X=X)

        assert len(y_pred) == 2

    def test_dml_forecaster_no_confounders(self):
        """Test DoubleMLForecaster when all variables are exposures (no confounders)."""
        y, X = load_longley()

        # Use all X columns as exposure variables
        exposure_vars = list(X.columns)

        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            exposure_vars=exposure_vars,
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        y_pred = forecaster.predict(fh=[1, 2], X=X)

        assert len(y_pred) == 2

    def test_dml_forecaster_fitted_params(self):
        """Test the get_fitted_params method."""
        y, X = load_longley()

        forecaster = DoubleMLForecaster(
            forecaster_y=NaiveForecaster(strategy="mean"),
            forecaster_ex=NaiveForecaster(strategy="mean"),
            forecaster_res=make_reduction(LinearRegression()),
            exposure_vars=["GNP"],
        )

        forecaster.fit(y, X=X, fh=[1, 2])
        fitted_params = forecaster.get_fitted_params()

        # Check that fitted parameters include residuals
        assert "y_residuals_" in fitted_params
        assert "X_ex_residuals_" in fitted_params
        assert "forecaster_y_final_" in fitted_params

    def test_check_estimator(self):
        """Test compatibility with sktime's check_estimator."""
        check_estimator(DoubleMLForecaster, verbose=False)


@pytest.mark.parametrize("parameter_set", ["default", "complex"])
def test_dml_forecaster_parameter_sets(parameter_set):
    """Test different parameter combinations."""
    params = DoubleMLForecaster.get_test_params(parameter_set)

    if isinstance(params, list):
        for param_dict in params:
            forecaster = DoubleMLForecaster(**param_dict)
            # Basic smoke test
            assert forecaster is not None
            assert hasattr(forecaster, "exposure_vars")
            assert len(forecaster.exposure_vars) > 0
    else:
        forecaster = DoubleMLForecaster(**params)
        assert forecaster is not None
        assert hasattr(forecaster, "exposure_vars")
        assert len(forecaster.exposure_vars) > 0


def test_dml_forecaster_get_test_params():
    """Test that get_test_params returns valid parameters."""
    params_default = DoubleMLForecaster.get_test_params("default")
    params_complex = DoubleMLForecaster.get_test_params("complex")

    # Should return a list of parameter dictionaries
    assert isinstance(params_default, list)
    assert isinstance(params_complex, list)
    assert len(params_default) > 0
    assert len(params_complex) > 0

    # Each parameter set should contain required keys
    for params in params_default + params_complex:
        assert "forecaster_y" in params
        assert "forecaster_ex" in params
        assert "exposure_vars" in params
        assert isinstance(params["exposure_vars"], list)
        assert len(params["exposure_vars"]) > 0


def test_dml_forecaster_tag_inheritance():
    """Tests that DoubleMLForecaster properly inherits tags from components."""
    # Create forecasters with different capabilities
    forecaster_with_intervals = NaiveForecaster(strategy="mean")
    forecaster_without_intervals = make_reduction(LinearRegression())

    # Test tag inheritance for prediction intervals
    dml_with_intervals = DoubleMLForecaster(
        forecaster_y=forecaster_with_intervals,
        forecaster_ex=forecaster_with_intervals,
        forecaster_res=forecaster_with_intervals,
        exposure_vars=["var_0"],
    )

    dml_without_intervals = DoubleMLForecaster(
        forecaster_y=forecaster_without_intervals,
        forecaster_ex=forecaster_without_intervals,
        forecaster_res=forecaster_without_intervals,
        exposure_vars=["var_0"],
    )

    # Check that tags are properly inherited
    assert dml_with_intervals.get_tag("capability:pred_int")
    assert not dml_without_intervals.get_tag("capability:pred_int")
