import pytest
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
.
# from sktime.forecasting.pytorch_forecasting import DeepARF 


def test_torch_2_6_checkpoint_loading_fix():
    """
    Checks if model cloning (which involves checkpoint loading) works 
    without UnpicklingError in PyTorch >= 2.6 due to the safe_globals fix.
    """
    
    try:
        from sktime.forecasting.pytorch_forecasting import DeepARF
        model = DeepARF(max_epochs=1) 
    except ImportError:
        pytest.skip("DeepARF (pytorch-forecasting dependency) not available for testing.")
        return

    y = pd.Series(np.random.rand(50), index=pd.RangeIndex(50), name="y")
    y_train, _ = temporal_train_test_split(y)
    fh = ForecastingHorizon([1, 2])

    try:
        model.fit(y=y_train, fh=fh)
    except Exception as e:
        pytest.fail(f"Model fitting failed unexpectedly: {e}")

    try:
        cloned_model = model.clone() 
    except Exception as e:
        pytest.fail(
            f"Model cloning (checkpoint loading) failed. The safe_globals fix may not be working. Error: {e}"
        )

    assert isinstance(cloned_model, DeepARF)
    assert cloned_model.is_fitted
