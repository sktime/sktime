import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pytest

from sktime.datasets import load_shampoo_sales
from sktime.forecasters import DummyForecaster
from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
from sktime.highlevel.tasks import ForecastingTask
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser
from sktime.utils.validation import validate_fh
from sktime.utils.transformations import select_times


regressor = Pipeline([('tabularise', Tabulariser()), ('clf', RandomForestRegressor(n_estimators=2))])


# Test forecasting strategy
@pytest.mark.parametrize("dynamic", [True, False])
@pytest.mark.parametrize("fh", [1, np.arange(1, 4)])
def test_univariate(dynamic, fh):

    fh = validate_fh(fh)
    n_fh = len(fh)

    y = load_shampoo_sales(return_y_as_dataframe=True)

    index = np.arange(y.iloc[0, 0].shape[0])
    train_times = index[:-n_fh]
    test_times = index[-n_fh:]

    y_train = select_times(y, train_times)
    y_test = select_times(y, test_times)

    task = ForecastingTask(target="ShampooSales", fh=fh, metadata=y_train)

    s = Forecasting2TSRReductionStrategy(estimator=regressor, dynamic=dynamic)
    s.fit(task, y_train)
    y_pred = s.predict()
    assert y_pred.shape == y_test[task.target].iloc[0].shape

