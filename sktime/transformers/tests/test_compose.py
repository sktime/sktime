import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

from sktime.pipeline import Pipeline
from sktime.tests.test_pipeline import X_train, y_train, X_test, y_test
from sktime.transformers.compose import ColumnTransformer, Tabulariser, RowwiseTransformer
from sktime.datasets import load_gunpoint

# load data
X_train, y_train = load_gunpoint("TRAIN", return_X_y=True)
X_train = pd.concat([X_train, X_train], axis=1)
X_train.columns = ['ts', 'ts_copy']

X_test, y_test = load_gunpoint("TEST", return_X_y=True)
X_test = pd.concat([X_test, X_test], axis=1)
X_test.columns = ['ts', 'ts_copy']


def test_ColumnTransformer_pipeline():
    # using Identity function transformers (transform series to series)
    id_func = lambda X: X
    column_transformer = ColumnTransformer(
        [('ts', FunctionTransformer(func=id_func, validate=False), 'ts'),
         ('ts_copy', FunctionTransformer(func=id_func, validate=False), 'ts_copy')])
    steps = [
        ('feature_extract', column_transformer),
        ('tabularise', Tabulariser()),
        ('rfestimator', RandomForestClassifier(n_estimators=2))]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))


def test_RowwiseTransformer_pipeline():
    # using pure sklearn
    mean_func = lambda X: pd.DataFrame([np.mean(row) for row in X])
    first_func = lambda X: pd.DataFrame([row[0] for row in X])
    column_transformer = ColumnTransformer(
        [('mean', FunctionTransformer(func=mean_func, validate=False), 'ts'),
         ('first', FunctionTransformer(func=first_func, validate=False), 'ts_copy')])
    estimator = RandomForestClassifier(n_estimators=2, random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(steps=strategy)
    model.fit(X_train, y_train)
    expected = model.predict(X_test)

    # using sktime with sklearn pipeline
    first_func = lambda X: pd.DataFrame([row[0] for row in X])
    column_transformer = ColumnTransformer(
        [('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False)), 'ts'),
         ('first', FunctionTransformer(func=first_func, validate=False), 'ts_copy')])
    estimator = RandomForestClassifier(n_estimators=2, random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = Pipeline(steps=strategy)
    model.fit(X_train, y_train)
    got = model.predict(X_test)
    np.testing.assert_array_equal(expected, got)