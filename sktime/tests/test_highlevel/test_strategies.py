import pytest
import pandas as pd
import numpy as np

from sktime.highlevel import TSCStrategy
from sktime.highlevel import ForecastingStrategy
from sktime.highlevel import ReduceForecasting2TSRStrategy
from sktime.highlevel import TSCTask
from sktime.highlevel import ForecastingTask

from sktime.datasets import load_gunpoint
from sktime.datasets import load_italy_power_demand
from sktime.datasets import load_shampoo_sales

from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.pipeline import TSPipeline
from sktime.transformers.compose import Tabulariser
from sktime.forecasting.forecasters import DummyForecaster

from sklearn.ensemble import RandomForestRegressor


classifier = TimeSeriesForestClassifier(n_estimators=2)
regressor = TSPipeline([('tabularise', Tabulariser()), ('clf', RandomForestRegressor(n_estimators=2))])
forecaster = DummyForecaster()

DATASET_LOADERS = (load_gunpoint, load_italy_power_demand)



# Test output of time-series classification strategies
@pytest.mark.parametrize("dataset", DATASET_LOADERS)
def test_TSCStrategy(dataset):
    train = dataset(split='TRAIN')
    test = dataset(split='TEST')
    s = TSCStrategy(classifier)
    task = TSCTask(target='class_val')
    s.fit(task, train)
    y_pred = s.predict(test)
    assert y_pred.shape == test[task.target].shape


# Test forecasting strategy
def test_ForecastingStrategy():
    shampoo = load_shampoo_sales(return_dataframe=True)
    train = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:30]]), columns=shampoo.columns)
    test = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[30:]]), columns=shampoo.columns)

    target = "ShampooSales"
    fh = np.arange(len(test[target].iloc[0])) + 1
    task = ForecastingTask(target=target, fh=fh, metadata=train)

    s = ForecastingStrategy(estimator=forecaster)
    s.fit(task, train)
    y_pred = s.predict()
    assert y_pred.shape == test[task.target].iloc[0].shape


# Test forecasting strategy
def test_Forecasting2TSRReductionStrategy():
    shampoo = load_shampoo_sales(return_dataframe=True)
    train = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:30]]), columns=shampoo.columns)
    test = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[30:]]), columns=shampoo.columns)

    target = "ShampooSales"
    fh = np.arange(len(test[target].iloc[0])) + 1
    task = ForecastingTask(target=target, fh=fh, metadata=train)

    s = ReduceForecasting2TSRStrategy(estimator=regressor)
    s.fit(task, train)
    y_pred = s.predict()
    assert y_pred.shape == test[task.target].iloc[0].shape
