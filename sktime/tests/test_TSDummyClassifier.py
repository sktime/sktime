from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from xpandas.data_container import XSeries, XDataFrame

from sklearn.utils.testing import assert_array_equal

from sktime import TSDummyClassifier


def read_data(file):
    data = file.readlines()
    rows = [row.decode('utf-8').strip().split('  ') for row in data]
    X = pd.DataFrame(rows, dtype=np.float)
    y = X.pop(0)
    return X, y

url = 'http://www.timeseriesclassification.com/Downloads/GunPoint.zip'
url = urlopen(url)
zipfile = ZipFile(BytesIO(url.read()))

train_file = zipfile.open('GunPoint_TRAIN.txt')
X_train_pd, y_train_pd = read_data(train_file)

test_file = zipfile.open('GunPoint_TEST.txt')
X_test_pd, y_test_pd = read_data(test_file)

y_train = XSeries(np.array(y_train_pd, dtype=np.int))
Xsf_train = XSeries([row for _, row in X_train_pd.iterrows()])
Xdf_train = XDataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})

def test_series_TSDummyClassifier_most_strategy():
    X = Xsf_train
    y = y_train
    model = TSDummyClassifier(strategy='most')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*2)

def test_dataframe_TSDummyClassifier_most_strategy():
    X = Xdf_train
    y = y_train
    model = TSDummyClassifier(strategy='most')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*2)

def test_series_TSDummyClassifier_least_strategy():
    X = Xsf_train
    y = y_train
    model = TSDummyClassifier(strategy='least')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*1)

def test_dataframe_TSDummyClassifier_least_strategy():
    X = Xdf_train
    y = y_train
    model = TSDummyClassifier(strategy='least')
    model.fit(X, y)
    preds = model.predict(X)
    assert_array_equal(preds, np.ones(X.shape[0])*1)
