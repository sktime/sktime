from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pytest
import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_array_equal

from sktime.classifiers.example_classifiers import TSDummyClassifier
from sktime.datasets import load_gunpoint

Xsf_train, y_train = load_gunpoint()
Xdf_train = pd.DataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})
Xsf_test, y_test = load_gunpoint("TEST")
Xdf_test = pd.DataFrame({'ts': Xsf_test, 'ts_copy': Xsf_test})


def read_data(file):
    '''
    adhoc function to read data
    '''
    data = file.readlines()
    rows = [row.decode('utf-8').strip().split('  ') for row in data]
    X = pd.DataFrame(rows, dtype=np.float)
    y = X.pop(0)
    return X, y

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

def test_dataframe_TSDummyClassifier_error_strategy():
    X = Xdf_train
    y = y_train
    with pytest.raises(ValueError, match="Unknown Strategy"):
        model = TSDummyClassifier(strategy='magic')
        model.fit(X, y)
