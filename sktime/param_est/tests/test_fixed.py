import pandas as pd

from sktime.param_est.fixed import FixedParams


def test_fixed_params():
    """Test FixedParams sets parameters correctly."""
    X = pd.Series([1, 2, 3])
    
    # Test with string keys and different types
    param_dict = {"alpha": 0.5, "beta": "string_val", "gamma": [1, 2, 3]}
    estimator = FixedParams(param_dict)
    
    estimator.fit(X)
    
    # Verify attributes end with "_"
    assert estimator.alpha_ == 0.5
    assert estimator.beta_ == "string_val"
    assert estimator.gamma_ == [1, 2, 3]
    
    # Verify get_fitted_params() works correctly
    fitted_params = estimator.get_fitted_params()
    assert fitted_params["alpha"] == 0.5
    assert fitted_params["beta"] == "string_val"
    assert fitted_params["gamma"] == [1, 2, 3]


def test_fixed_params_empty():
    """Test FixedParams with empty dictionary."""
    X = pd.Series([1, 2, 3])
    estimator = FixedParams({})
    
    estimator.fit(X)
    
    # Should not raise an error, get_fitted_params should be empty
    assert estimator.get_fitted_params() == {}
