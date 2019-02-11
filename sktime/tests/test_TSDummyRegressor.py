from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pytest
import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_array_equal

from sktime.regressors import TSDummyRegressor


def read_data(file):
    '''
    adhoc function to read data
    '''
    data = file.readlines()
    rows = [row.decode('utf-8').strip().split('  ') for row in data]
    X = pd.DataFrame(rows, dtype=np.float)
    y = X.pop(0)
    return X, y

# For simplicity, the classification labels are used as regression targets for testing
url = 'http://www.timeseriesclassification.com/Downloads/GunPoint.zip'
url = urlopen(url)
zipfile = ZipFile(BytesIO(url.read()))

train_file = zipfile.open('GunPoint_TRAIN.txt')
X_train_pd, y_train_pd = read_data(train_file)

test_file = zipfile.open('GunPoint_TEST.txt')
X_test_pd, y_test_pd = read_data(test_file)

y_train = pd.Series(np.array(y_train_pd, dtype=np.int))
Xsf_train = pd.Series([row for _, row in X_train_pd.iterrows()])
Xdf_train = pd.DataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})

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
