"""Tests for WindFMForecaster."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.forecasting.windfm import WindFMForecaster


def test_windfm_predict_proba_preserves_sample_paths():
    """Empirical forecast should contain the generated WindFM sample paths."""
    pytest.importorskip("skpro")

    y_index = pd.Index([10, 11], name="time")
    pred_df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=y_index,
        columns=["sample_0", "sample_1", "sample_2"],
    )
    forecaster = WindFMForecaster()
    forecaster._state = "fitted"
    forecaster._is_vectorized = False
    forecaster._y_metadata = {"feature_names": ["power"]}
    forecaster._set_cutoff(9)
    forecaster._predict_samples = lambda fh: (pred_df, y_index)

    pred_dist = forecaster.predict_proba(fh=[1, 2], marginal=False)

    expected_samples = pd.DataFrame(
        [[1.0], [4.0], [2.0], [5.0], [3.0], [6.0]],
        index=pd.MultiIndex.from_product(
            [pred_df.columns, y_index], names=["sample", "time"]
        ),
        columns=["power"],
    )
    expected_quantiles = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=y_index,
        columns=pd.MultiIndex.from_product([["power"], [0.1, 0.5, 0.9]]),
    )
    proba_quantiles = pred_dist.quantile([0.1, 0.5, 0.9])
    direct_quantiles = forecaster.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.5, 0.9])

    assert_frame_equal(pred_dist.spl, expected_samples)
    assert not pred_dist.time_indep
    assert_frame_equal(proba_quantiles, expected_quantiles)
    assert_frame_equal(direct_quantiles, expected_quantiles)
