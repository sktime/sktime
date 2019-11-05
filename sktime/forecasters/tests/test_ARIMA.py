import pytest
import numpy as np
import pandas as pd

from sktime.forecasters import ARIMAForecaster
from sktime.datasets import load_shampoo_sales
from sktime.datasets import load_longley
from sktime.forecasters.model_selection import temporal_train_test_split

__author__ = "Markus Löning"


# forecast horizons
FHS = ([1], np.arange(1, 5))


# TODO currently tests only run fit/predict and compare length of predicted series
@pytest.mark.parametrize("fh", FHS)
def test_ARIMAForecaster_univariate(fh):
    y = load_shampoo_sales()
    y_train, y_test = temporal_train_test_split(y, fh)

    m = ARIMAForecaster()
    m.fit(y_train)
    y_pred = m.predict(fh=fh)
    assert y_pred.shape[0] == len(fh)
    assert m.score(y_test, fh=fh) > 0


@pytest.mark.parametrize("fh", FHS)
def test_ARIMAForecaster_multivariate(fh):
    X, y = load_longley(return_X_y=True)

    #  get data in required format
    y_train, y_test = temporal_train_test_split(y, fh)

    max_fh = np.max(fh)
    m = len(y)
    cutoff = m - max_fh
    X_train = pd.DataFrame([pd.Series([X.iloc[0, i].iloc[:cutoff]])
                            for i in range(X.shape[1])]).T
    X_train.columns = X.columns
    X_test = pd.DataFrame([pd.Series([X.iloc[0, i].iloc[cutoff:]])
                           for i in range(X.shape[1])]).T
    X_test.columns = X.columns

    m = ARIMAForecaster()
    m.fit(y_train, X=X_train)
    y_pred = m.predict(fh=fh, X=X_test)
    assert y_pred.shape[0] == len(fh)
    assert m.score(y_test, fh=fh, X=X_test) > 0
