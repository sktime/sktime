"""Check the actual training rows of global direct reduction models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from sktime.forecasting.compose import make_reduction
from sktime.transformations.summarize import WindowSummarizer


class _RecordingRegressor(RegressorMixin, BaseEstimator):
    """Retain the data passed to fit, independently of reduction internals."""

    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize("hierarchy_levels", [0, 1, 2])
@pytest.mark.parametrize("windows_identical", [False, True])
@pytest.mark.parametrize("with_exogenous", [False, True])
def test_direct_global_training_rows(
    hierarchy_levels, windows_identical, with_exogenous
):
    """Horizon h pairs features at t with targets at t+h-1 in each series."""
    n_timepoints = 12
    window_length = 3
    fh = [1, 2, 5]
    if hierarchy_levels:
        index = pd.MultiIndex.from_product(
            [["a", "b"]] * hierarchy_levels + [range(n_timepoints)],
            names=[f"level{i}" for i in range(hierarchy_levels)] + ["time"],
        )
    else:
        index = pd.RangeIndex(n_timepoints, name="time")
    y = pd.DataFrame({"value": np.arange(len(index), dtype=float)}, index=index)
    X = y.rename(columns={"value": "exogenous"}) + 100 if with_exogenous else None

    forecaster = make_reduction(
        _RecordingRegressor(),
        strategy="direct",
        pooling="global",
        windows_identical=windows_identical,
        window_length=None,
        transformers=[WindowSummarizer(lag_feature={"lag": [1, 3]}, n_jobs=1)],
    ).fit(y, X=X, fh=fh)

    groups = (
        y.groupby(level=list(range(hierarchy_levels)))
        if hierarchy_levels
        else [(None, y)]
    )
    for horizon, regressor in zip(fh, forecaster.estimators_):
        expected_X = []
        expected_y = []
        for _, group in groups:
            # 特征使用 t 之前的值; 目标相对 t 按预测步数移动, 不能跨实例配对。
            stop = n_timepoints - (max(fh) if windows_identical else horizon) + 1
            times = np.arange(window_length, stop)
            values = group.iloc[:, 0].to_numpy()
            features = np.column_stack([values[times - 1], values[times - 3]])
            if with_exogenous:
                features = np.column_stack([features, values[times] + 100])
            expected_X.append(features)
            expected_y.append(values[times + horizon - 1])
        np.testing.assert_array_equal(
            regressor.X_.to_numpy(), np.concatenate(expected_X)
        )
        np.testing.assert_array_equal(
            np.asarray(regressor.y_).reshape(-1), np.concatenate(expected_y)
        )


def test_direct_global_forecasts_by_horizon():
    """A linear series extrapolates by the requested number of steps."""
    y = pd.Series(np.arange(12.0))
    forecaster = make_reduction(
        LinearRegression(),
        strategy="direct",
        pooling="global",
        windows_identical=False,
        window_length=3,
    ).fit(y, fh=[1, 3, 5])
    prediction = forecaster.predict()
    np.testing.assert_allclose(prediction.to_numpy(), [12.0, 14.0, 16.0])
    pd.testing.assert_index_equal(prediction.index, pd.Index([12, 14, 16]))


@pytest.mark.parametrize("windows_identical", [False, True])
def test_direct_global_insufficient_training_rows(windows_identical):
    """Reject horizons that leave no complete feature/target pair."""
    forecaster = make_reduction(
        _RecordingRegressor(),
        strategy="direct",
        pooling="global",
        windows_identical=windows_identical,
        window_length=3,
    )
    with pytest.raises(ValueError, match="window_length.*fh.*incompatible"):
        forecaster.fit(pd.Series(np.arange(12.0)), fh=[10])
