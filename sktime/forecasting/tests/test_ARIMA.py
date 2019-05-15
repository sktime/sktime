import pytest
import numpy as np
import pandas as pd

from sktime.forecasting.forecasters import ARIMAForecaster
from sktime.datasets import load_shampoo_sales
from sktime.datasets import load_longley

__author__ = "Markus Löning"


# forecast horizons
FHS = ([1], np.arange(1, 5))


# TODO currently tests only run fit/predict and compare length of predicted series
@pytest.mark.parametrize("fh", FHS)
def test_ARIMAForecaster_univariate(fh):
    y = load_shampoo_sales()

    max_fh = np.max(fh)
    m = len(y.iloc[0])
    cutoff = m - max_fh

    y_train = pd.Series([y.iloc[0].iloc[:cutoff]])
    y_test = pd.Series([y.iloc[0].iloc[cutoff:]])

    m = ARIMAForecaster()
    m.fit(y_train)
    y_pred = m.predict(fh=fh)
    assert y_pred.shape[0] == len(fh)
    assert m.score(y_test, fh=fh) > 0


@pytest.mark.parametrize("fh", FHS)
def test_ARIMAForecaster_multivariate(fh):
    X, y = load_longley(return_X_y=True)

    #  get data in required format
    max_fh = np.max(fh)
    m = len(y.iloc[0])
    cutoff = m - max_fh

    y_train = pd.Series([y.iloc[0].iloc[:cutoff]])
    y_test = pd.Series([y.iloc[0].iloc[cutoff:]])
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
