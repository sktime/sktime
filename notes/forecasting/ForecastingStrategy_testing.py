import numpy as np
import pandas as pd

from sktime.datasets import load_longley
from sktime.datasets import load_shampoo_sales
from sktime.forecasting import DummyForecaster
from sktime.highlevel.strategies import ForecastingStrategy
from sktime.highlevel.tasks import ForecastingTask

forecaster = DummyForecaster()


# Test forecasting strategy
def test_univariate():
    y = load_shampoo_sales()
    target = "ShampooSales"
    y.name = target
    train = pd.DataFrame(pd.Series([y.iloc[:30]]), columns=[target])
    test = pd.DataFrame(pd.Series([y.iloc[30:]]), columns=[target])

    fh = np.arange(len(test.loc[:, target].iloc[0])) + 1
    task = ForecastingTask(target=target, fh=fh, metadata=train)

    s = ForecastingStrategy(estimator=forecaster)
    s.fit(task, train)
    y_pred = s.predict()
    assert y_pred.shape == test[task.target].iloc[0].shape


def test_multivariate():
    longley = load_longley(return_X_y=False)
    train = pd.DataFrame([pd.Series([longley.iloc[0, i].iloc[:13]]) for i in range(longley.shape[1])]).T
    train.columns = longley.columns

    test = pd.DataFrame([pd.Series([longley.iloc[0, i].iloc[13:]]) for i in range(longley.shape[1])]).T
    test.columns = longley.columns
    target = "TOTEMP"
    fh = np.arange(len(test[target].iloc[0])) + 1
    task = ForecastingTask(target=target, fh=fh, metadata=train)

    estimator = ARIMAForecaster()
    s = ForecastingStrategy(estimator=estimator)
    s.fit(task, train)
    y_pred = s.predict(data=test)
    assert y_pred.shape == test[task.target].iloc[0].shape
