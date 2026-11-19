import numpy as np
import pandas as pd
import pytest

from sktime.transformations.subsequence_extraction import (
    SubsequenceExtractionTransformer,
)


@pytest.mark.parametrize("selector", ["max", "min"])
def test_mean_rolling_selection(selector):
    """Select the correct subsequence using the rolling mean."""
    values = [4.0, 1.0, 7.0, 2.0, 9.0, 3.0, 6.0, 8.0]
    X = pd.DataFrame({"y": values})

    transformer = SubsequenceExtractionTransformer(
        aggregate_fn=np.mean,
        subseq_len=3,
        selector=selector,
    )

    result = transformer.fit_transform(X)
    rolling_mean = X.rolling(window=3).mean()["y"]
    best_end = getattr(rolling_mean, f"idx{selector}")()
    expected = X.iloc[best_end - 2 : best_end + 1].reset_index(drop=True)

    pd.testing.assert_frame_equal(result, expected)
