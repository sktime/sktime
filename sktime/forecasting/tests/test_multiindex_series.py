import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

def test_multiindex_series_graceful_fit():
    """Test that a MultiIndexed pd.Series is gracefully promoted to DataFrame in fit."""
    # Issue #7780 verification dataset
    past_dataset = pd.DataFrame(
        {
            "series_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    ).set_index(["series_id", "time_id"])
    
    # Extracting as a MultiIndexed Series (Single Bracket)
    series_y = past_dataset["y"]
    
    model = NaiveForecaster()
    
    # This should seamlessly execute without throwing TypeError
    model.fit(y=series_y)
    
    # Sanity confirmation that the model successfully reached fitted state
    assert model.is_fitted == True