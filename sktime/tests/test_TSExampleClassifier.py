from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pytest
import numpy as np
import pandas as pd
from xpandas.data_container import XSeries, XDataFrame

from sklearn.utils.testing import assert_array_equal

from sktime.classifiers import TSExampleClassifier
from sklearn.ensemble import RandomForestClassifier


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
Xsf_test = XSeries([row for _, row in X_test_pd.iterrows()])
Xdf_test = XDataFrame({'ts': Xsf_test, 'ts_copy': Xsf_test})

y_train = XSeries(np.array(y_train_pd, dtype=np.int))
Xsf_train = XSeries([row for _, row in X_train_pd.iterrows()])
Xdf_train = XDataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})

def test_xdataframe_TSExampleClassifier():
    X = Xdf_train
    y = y_train
    model = TSExampleClassifier(func=np.mean, columns=X.columns, estimator=RandomForestClassifier(random_state=123, n_estimators=10))
    model.fit(X, y)
    assert_array_equal(model.predict(Xdf_test), np.ones(y_test_pd.shape[0]) * 2)

def test_set_get_param():
    X = Xdf_train
    y = y_train
    model = TSExampleClassifier(func=np.mean, columns=X.columns, estimator=RandomForestClassifier(random_state=123, n_estimators=10))
    model.set_params(estimator__random_state=42)
    assert model.get_params()['estimator__random_state'] == 42
