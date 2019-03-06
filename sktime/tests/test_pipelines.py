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
from sktime.datasets import load_gunpoint

Xsf_train, y_train = load_gunpoint()
Xdf_train = pd.DataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})
Xsf_test, y_test = load_gunpoint("TEST")
Xdf_test = pd.DataFrame({'ts': Xsf_test, 'ts_copy': Xsf_test})


def test_pipeline():
    X = Xdf_train
    y = y_train

    # using pure sklearn
    mean_func = lambda X: pd.DataFrame([np.mean(row) for row in X])
    first_func = lambda X: pd.DataFrame([row[0] for row in X])
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
    assert_array_equal(model.predict(Xdf_test), np.ones(y_test.shape[0]) * 2)


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
    assert_array_equal(model.predict(Xdf_test), np.ones(y_test.shape[0]) * 2)


def test_RowwiseTransformer_pipeline():
    X = Xdf_train
    y = y_train

    # using pure sklearn
    mean_func = lambda X: pd.DataFrame([np.mean(row) for row in X])
    first_func = lambda X: pd.DataFrame([row[0] for row in X])
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
    first_func = lambda X: pd.DataFrame([row[0] for row in X])
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
