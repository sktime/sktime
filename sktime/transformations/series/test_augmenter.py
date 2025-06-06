import numpy as np
import pandas as pd
import pytest
from sktime.transformations.series.augmenter import WhiteNoiseAugmenter

@pytest.mark.parametrize(
    "input_data",
    [
        np.array([1.0, 2.0, 3.0, 4.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4, freq="D")),
        pd.DataFrame(
            np.array([[1.0], [2.0], [3.0], [4.0]]),
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
            columns=["A"],
        ),
    ],
)
def test_white_noise_augmenter_shapes(input_data):
    augmenter = WhiteNoiseAugmenter(scale=0.1, random_state=0)
    # fit() should accept any of the above
    augmenter.fit(input_data)
    out = augmenter.transform(input_data)
    # Output type should match input type
    if isinstance(input_data, np.ndarray):
        assert isinstance(out, np.ndarray)
        assert out.shape == input_data.shape
    elif isinstance(input_data, pd.Series):
        assert isinstance(out, pd.Series)
        assert out.shape == input_data.shape
        assert out.index.equals(input_data.index)
    else:
        assert isinstance(out, pd.DataFrame)
        assert out.shape == input_data.shape
        assert out.index.equals(input_data.index)
        assert list(out.columns) == list(input_data.columns)

    # Values should differ (because noise > 0)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, input_data.values, out.values
    )
