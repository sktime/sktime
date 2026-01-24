import pandas as pd
import pytest
from sktime.transformations.series.difference import Differencer

def test_differencer_clipping():
    """Test that Differencer clips values correctly with min_value and max_value."""
    # Data: [10, 20, 30, 200, -100]
    # Diff: [NaN, 10, 10, 170, -300]
    X = pd.DataFrame({"a": [10, 20, 30, 200, -100]})
    
    # 1. Test MAX clipping (Limit to 20)
    # The 170 should become 20.
    transformer_max = Differencer(lags=1, max_value=20)
    Xt_max = transformer_max.fit_transform(X)
    assert Xt_max["a"].max() <= 20.0, "❌ Error: Max value was not clipped!"

    # 2. Test MIN clipping (Limit to -50)
    # The -300 should become -50.
    transformer_min = Differencer(lags=1, min_value=-50)
    Xt_min = transformer_min.fit_transform(X)
    assert Xt_min["a"].min() >= -50.0, "❌ Error: Min value was not clipped!"
    
    # 3. Test BOTH
    transformer_both = Differencer(lags=1, min_value=-5, max_value=5)
    Xt_both = transformer_both.fit_transform(X)
    assert Xt_both["a"].max() <= 5.0
    assert Xt_both["a"].min() >= -5.0

    print("✅ All Clipping Tests Passed!")

if __name__ == "__main__":
    test_differencer_clipping()