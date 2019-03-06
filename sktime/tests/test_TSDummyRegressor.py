from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pytest
import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_array_equal

from sktime.regressors.example_regressors import TSDummyRegressor
from sktime.datasets import load_gunpoint

Xsf_train, y_train = load_gunpoint()
Xdf_train = pd.DataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})
Xsf_test, y_test = load_gunpoint("TEST")
Xdf_test = pd.DataFrame({'ts': Xsf_test, 'ts_copy': Xsf_test})

def test_dataframe_TSDummyRegressor_constant_strategy():
    X = Xdf_train
    y = y_train
    model = TSDummyRegressor(strategy='constant')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*42)

def test_dataframe_TSDummyRegressor_average_strategy():
    X = Xdf_train
    y = y_train
    model = TSDummyRegressor(strategy='average')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*np.mean(y_train))

def test_series_TSDummyRegressor_constant_strategy():
    X = Xsf_train
    y = y_train
    model = TSDummyRegressor(strategy='constant')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*42)

def test_series_TSDummyRegressor_average_strategy():
    X = Xsf_train
    y = y_train
    model = TSDummyRegressor(strategy='average')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*np.mean(y_train))

def test_dataframe_TSDummyClassifier_error_strategy():
    X = Xdf_train
    y = y_train
    with pytest.raises(ValueError, match="Unknown Strategy"):
        model = TSDummyRegressor(strategy='magic')
        model.fit(X, y)
