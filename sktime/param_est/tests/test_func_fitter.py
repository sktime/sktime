import numpy as np
import pandas as pd
import pytest

from sktime.param_est.compose import FunctionParamFitter


def _calc_mean(X):
    return float(np.mean(X))


def _calc_quantile(X, q=0.5):
    return float(np.quantile(X, q=q))


def test_function_param_fitter_basic():
    """Test FunctionParamFitter sets parameter and trailing underscore correctly."""
    X = pd.Series([10.0, 20.0, 30.0])
    # Test with parameter name without trailing underscore
    fitter = FunctionParamFitter(param="mean_val", func=_calc_mean)
    fitter.fit(X)

    assert hasattr(fitter, "mean_val_")
    assert fitter.mean_val_ == 20.0
    assert fitter.get_fitted_params() == {"mean_val": 20.0}


def test_function_param_fitter_with_kwargs():
    """Test FunctionParamFitter forwards keyword arguments to the callable."""
    X = pd.Series([10.0, 20.0, 30.0, 40.0])
    # Test with parameter name with trailing underscore
    fitter = FunctionParamFitter(
        param="q75_",
        func=_calc_quantile,
        kw_args={"q": 0.75},
    )
    fitter.fit(X)

    assert hasattr(fitter, "q75_")
    assert fitter.q75_ == 32.5
    assert fitter.get_fitted_params() == {"q75": 32.5}


def test_function_param_fitter_type_conversion():
    """Test FunctionParamFitter converts input to specified X_type."""
    def _func_numpy(X):
        assert isinstance(X, np.ndarray)
        return int(np.sum(X))

    X_series = pd.Series([1, 2, 3, 4])
    fitter = FunctionParamFitter(
        param="sum_val",
        func=_func_numpy,
        X_type="np.ndarray",
    )
    fitter.fit(X_series)

    assert fitter.sum_val_ == 10
    assert fitter.get_fitted_params() == {"sum_val": 10}
