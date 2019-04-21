import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sktime.pipeline import TSPipeline
from sktime.transformers.compose import TSColumnTransformer
from sktime.transformers.compose import RowwiseTransformer
from sktime.datasets import load_gunpoint
from sktime.transformers.compose import Tabulariser
from sktime.pipeline import TSFeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sktime.transformers.series_to_series import RandomIntervalSegmenter
from sktime.classifiers.ensemble import TimeSeriesForestClassifier

# load data
X_train, y_train = load_gunpoint("TRAIN", return_X_y=True)
X_train = pd.concat([X_train, X_train], axis=1)
X_train.columns = ['ts', 'ts_copy']

X_test, y_test = load_gunpoint("TEST", return_X_y=True)
X_test = pd.concat([X_test, X_test], axis=1)
X_test.columns = ['ts', 'ts_copy']


def test_TSColumnTransformer_pipeline():
    """
    there is a series to series transformer tested in here
    """
    # using Identity function transformers (transform series to series)
    id_func = lambda X: X
    column_transformer = TSColumnTransformer(
        [('ts', FunctionTransformer(func=id_func, validate=False), 'ts'),
         ('ts_copy', FunctionTransformer(func=id_func, validate=False), 'ts_copy')])
    steps = [
        ('feature_extract', column_transformer),
        ('tabularise', Tabulariser()),
        ('rfestimator', RandomForestClassifier(n_estimators=3))]
    model = TSPipeline(steps=steps)
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
    estimator = RandomForestClassifier(random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = TSPipeline(steps=strategy)
    model.fit(X_train, y_train)
    expected = model.predict(X_test)

    # using sktime with sklearn pipeline
    first_func = lambda X: pd.DataFrame([row[0] for row in X])
    column_transformer = TSColumnTransformer(
        [('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False)), 'ts'),
         ('first', FunctionTransformer(func=first_func, validate=False), 'ts_copy')])
    estimator = RandomForestClassifier(random_state=1)
    strategy = [
        ('feature_extract', column_transformer),
        ('rfestimator', estimator)]
    model = TSPipeline(steps=strategy)
    model.fit(X_train, y_train)
    got = model.predict(X_test)
    np.testing.assert_array_equal(expected, got)


def test_TSFeatureUnion_pipeline():
    # pipeline with segmentation plus multiple feature extraction
    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals=3, check_input=False)),
        ('transform', TSFeatureUnion([
            ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
            ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False)))
        ])),
        ('clf', DecisionTreeClassifier())
    ]
    clf = TSPipeline(steps)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))
