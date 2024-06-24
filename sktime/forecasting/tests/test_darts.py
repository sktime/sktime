"""Test for HolidayFeatures transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

import re

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.darts import DartsXGBModel
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

__author__ = ["fnhirwa"]


y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)


@pytest.mark.skipif(
    not run_test_for_class(DartsXGBModel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_xgb_model_without_X():
    """Test with single endogenous without exogenous."""
    sktime_model = DartsXGBModel(
        lags=6,
        output_chunk_length=4,
    )
    # try to fit with negative forecast horizon (insample prediction)
    with pytest.raises(
        NotImplementedError, match="in-sample prediction is currently not supported"
    ):
        sktime_model.fit(y_train, fh=[-2, -1, 0, 1, 2])

    # train the model
    sktime_model.fit(y_train, fh=[1, 2, 3, 4])
    # make prediction
    pred = sktime_model.predict()

    # check the index of the prediction
    pd.testing.assert_index_equal(pred.index, y_test.index, check_names=False)


@pytest.mark.skipif(
    not run_test_for_class(DartsXGBModel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_darts_xgb_model_with_weather_dataset():
    """Test with weather dataset."""
    from darts.datasets import WeatherDataset
    from darts.models import XGBModel

    # Load the dataset
    series = WeatherDataset().load()

    # Predicting atmospheric pressure
    target = series["p (mbar)"][:100]
    target_df = target.pd_series()
    # Create and fit the model
    darts_model = XGBModel(
        lags=12,
        output_chunk_length=6,
    )

    darts_model.fit(target)

    # Make a prediction for the next 6 time steps
    darts_pred = darts_model.predict(6).pd_series()
    assert isinstance(target_df, pd.Series)
    sktime_model = DartsXGBModel(
        lags=12,
        output_chunk_length=6,
    )
    sktime_model.fit(target_df)
    fh = list(range(1, 7))
    pred_sktime = sktime_model.predict(fh)
    assert isinstance(pred_sktime, pd.Series)

    np.testing.assert_array_equal(pred_sktime.to_numpy(), darts_pred.to_numpy())


# @pytest.mark.skipif(
#     not run_test_for_class(DartsXGBModel),
#     reason="run test only if softdeps are present and incrementally (if requested)",
# )
def test_darts_xgb_model_with_X():
    """Test with single endogenous and exogenous."""
    past_covariates = ["GNPDEFL", "GNP", "UNEMP"]
    sktime_model = DartsXGBModel(
        lags=6, output_chunk_length=4, past_covariates=["GNPDEFL", "GNP", "UNEMP"]
    )
    expected_message = re.escape(
        f"Expected following exogenous features: {past_covariates}."
    )
    # attempt fitting without exogenous
    with pytest.raises(ValueError, match=expected_message):
        sktime_model.fit(y_train, fh=[1, 2, 3, 4])

    sktime_model.fit(y_train, fh=[1, 2, 3, 4], X=X_train)
    # attempt to predict without exogenous
    with pytest.raises(ValueError, match=expected_message):
        sktime_model.predict()
    pred = sktime_model.predict(X=X_test[past_covariates])

    # check the index of the prediction
    pd.testing.assert_index_equal(pred.index, y_test.index, check_names=False)
