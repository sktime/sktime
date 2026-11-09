# sktime/forecasting/tests/test_causal_var.py

import pytest
import pandas as pd
from pgmpy.models import BayesianNetwork

# Important: Make sure to place your CausalVAR class in the right folder to import
from sktime.forecasting._causal_var import CausalVAR
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


def test_causal_var_end_to_end():
    """
    An end-to-end test for the CausalVAR forecaster.
    """
    # 1. Define a causal structure
    # 'sales' is caused by 'marketing', 'inventory' is caused by 'sales'
    causal_graph = BayesianNetwork(
        [("marketing", "sales"), ("sales", "inventory")]
    )

    # 2. Create sample multivariate time series data
    # This data loosely follows the causal structure for a plausible test
    idx = pd.to_datetime(pd.date_range("2023-01-01", periods=50, freq="D"))
    data = pd.DataFrame(
        {
            "marketing": pd.Series(range(50)) * 0.5 + 10,
            "sales": pd.Series(range(50)) * 2 + 5,
            "inventory": 100 - pd.Series(range(50)) * 1.5,
        },
        index=idx,
    )
    # Add some noise
    data += pd.np.random.rand(50, 3) * 5

    # 3. Split data for training and testing
    y_train, y_test = temporal_train_test_split(data, test_size=10)

    # 4. Instantiate and fit the forecaster
    forecaster = CausalVAR(causal_graph=causal_graph, maxlags=3)
    forecaster.fit(y_train)

    # 5. Make a prediction
    fh = y_test.index
    y_pred = forecaster.predict(fh)

    # 6. Assertions
    # Check that output is a pandas DataFrame
    assert isinstance(y_pred, pd.DataFrame)
    # Check that the dimensions are correct
    assert y_pred.shape == y_test.shape
    # Check that the index matches the forecast horizon
    assert all(y_pred.index == y_test.index)
    
    # Check that a reasonable forecast is made (not perfect, but not nonsensical)
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    assert mape < 0.5 # MAPE should be less than 50% for this simple data

    # Test error handling for bad graph
    bad_graph = BayesianNetwork([("non_existent_node", "sales")])
    with pytest.raises(ValueError):
        bad_forecaster = CausalVAR(bad_graph)
        bad_forecaster.fit(y_train)