# -*- coding: utf-8 -*-
from sktime.forecasting.var import VectorAutoRegression as VAR
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR as _VAR
from numpy.testing import assert_allclose

index = pd.date_range(start="2005", end="2006-12", freq="M")
df = pd.DataFrame(
    np.random.randint(0, 100, size=(23, 2)),
    columns=list("AB"),
    index=pd.PeriodIndex(index),
)


def test_var():
    train, test = temporal_train_test_split(df)
    sktime_model = VAR()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    sktime_model.fit(train)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VAR(train)
    stats_fit = stats.fit()
    fh_int = fh.to_absolute_int(train.index[0], train.index[-1])
    lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.forecast(train.values[-lagged:], steps=fh_int[-1])
    new_arr = []
    for i in fh:
        new_arr.append(y_pred_stats[i - 1])
    assert_allclose(y_pred, new_arr)
