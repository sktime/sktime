import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.split import temporal_train_test_split


def test_update_simple_model_matches_refit():
    """Test that update matches full refit for simple model."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)

    f = ExponentialSmoothing()
    f.fit(y_train)
    f.update(y_test.iloc[:3])

    f2 = ExponentialSmoothing()
    f2.fit(pd.concat([y_train, y_test.iloc[:3]]))

    pred1 = f.predict([1, 2, 3]).values
    pred2 = f2.predict([1, 2, 3]).values

    assert np.allclose(pred1, pred2)


def test_update_trend_seasonal_close_to_refit():
    """Test that update is close to refit for seasonal model."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)

    f = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
    f.fit(y_train)
    f.update(y_test.iloc[:3])

    f2 = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
    f2.fit(pd.concat([y_train, y_test.iloc[:3]]))

    pred1 = f.predict([1, 2, 3]).values
    pred2 = f2.predict([1, 2, 3]).values

    assert np.allclose(pred1, pred2, rtol=1e-1)