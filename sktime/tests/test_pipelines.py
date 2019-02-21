from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_array_equal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from sktime.classifiers.example_classifiers import TSExampleClassifier
from sktime.transformers.example_transformers import TSExampleTransformer, TSDummyTransformer
from sktime.transformers.compose import TSColumnTransformer
from sktime.transformers.compose import RowwiseTransformer


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
Xsf_test = pd.Series([row for _, row in X_test_pd.iterrows()])
Xdf_test = pd.DataFrame({'ts': Xsf_test, 'ts_copy': Xsf_test})

y_train = pd.Series(np.array(y_train_pd, dtype=np.int))
Xsf_train = pd.Series([row for _, row in X_train_pd.iterrows()])
Xdf_train = pd.DataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})


def test_pipeline():
    X = Xdf_train
    y = y_train

    # using pure sklearn
    mean_func = lambda X: pd.DataFrame([np.mean(row) for row in X])
    first_func = lambda X: pd.DataFrame([row.iloc[0] for row in X])
    column_transformer = ColumnTransformer(
        [('mean', FunctionTransformer(func=mean_func, validate=False), 'ts'),
         ('first', FunctionTransformer(func=first_func, validate=False), 'ts_copy')])
    estimator = RandomForestClassifier(random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(memory=None,
                     steps=strategy)
    model.fit(X, y)
    expected = model.predict(X)

    # using sktime with sklearn pipeline
    column_transformer = TSExampleTransformer()
    estimator = TSExampleClassifier(estimator=RandomForestClassifier(random_state=1))
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(memory=None,
                     steps=strategy)
    model.fit(X, y)
    got = model.predict(X)
    assert_array_equal(expected, got)


def test_series_pipeline():
    '''
    there is a series to series transformer tested in here
    '''
    X = Xdf_train
    y = y_train

    column_transformer = TSDummyTransformer()
    estimator = TSExampleClassifier(func=np.mean, columns=X.columns, estimator=RandomForestClassifier(random_state=123, n_estimators=10))
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(memory=None,
                     steps=strategy)
    model.fit(X, y)
    assert_array_equal(model.predict(Xdf_test), np.ones(y_test_pd.shape[0]) * 2)


def test_pandas_friendly_column_transformer_pipeline():
    '''
    there is a series to series transformer tested in here
    '''
    X = Xdf_train
    y = y_train

    estimator = TSExampleClassifier(func=np.mean, columns=X.columns, estimator=RandomForestClassifier(random_state=123, n_estimators=10))
    # using Identity function transformers (transform series to series)
    id_func = lambda X: X
    column_transformer = TSColumnTransformer(
        [('ts', FunctionTransformer(func=id_func, validate=False), 'ts'),
         ('ts_copy', FunctionTransformer(func=id_func, validate=False), 'ts_copy')])
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(memory=None,
                     steps=strategy)
    model.fit(X, y)
    assert_array_equal(model.predict(Xdf_test), np.ones(y_test_pd.shape[0]) * 2)


def test_RowwiseTransformer_pipeline():
    X = Xdf_train
    y = y_train

    # using pure sklearn
    mean_func = lambda X: pd.DataFrame([np.mean(row) for row in X])
    first_func = lambda X: pd.DataFrame([row.iloc[0] for row in X])
    column_transformer = ColumnTransformer(
        [('mean', FunctionTransformer(func=mean_func, validate=False), 'ts'),
         ('first', FunctionTransformer(func=first_func, validate=False), 'ts_copy')])
    estimator = RandomForestClassifier(random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(memory=None,
                     steps=strategy)
    model.fit(X, y)
    expected = model.predict(X)

    # using sktime with sklearn pipeline
    first_func = lambda X: pd.DataFrame([row.iloc[0] for row in X])
    column_transformer = TSColumnTransformer(
        [('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False)), 'ts'),
         ('first', FunctionTransformer(func=first_func, validate=False), 'ts_copy')])
    estimator = RandomForestClassifier(random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(memory=None,
                     steps=strategy)
    model.fit(X, y)
    got = model.predict(X)
    assert_array_equal(expected, got)
