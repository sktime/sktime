from ..classifiers.ensemble import TimeSeriesForestClassifier
from ..utils.testing import generate_df_from_array
import pandas as pd
import numpy as np
from sktime.transformers.compose import RowwiseTransformer
from sktime.datasets import load_gunpoint
from sktime.pipeline import TSFeatureUnion, TSPipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.transformers.series_to_series import RandomIntervalSegmenter
from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sklearn.preprocessing import FunctionTransformer
from sktime.utils.time_series import time_series_slope

N_ITER = 10

n = 20
d = 1
m = 20
n_classes = 2

X = generate_df_from_array(np.random.normal(size=m), n_rows=n, n_cols=d)
y = pd.Series(np.random.choice(np.arange(n_classes) + 1, size=n))


# Check if random state always gives same results
def test_random_state():
    random_state = 1234
    clf = TimeSeriesForestClassifier(n_estimators=2,
                                     random_state=random_state)
    clf.fit(X, y)
    first_pred = clf.predict_proba(X)
    for _ in range(N_ITER):
        clf = TimeSeriesForestClassifier(n_estimators=2,
                                         random_state=random_state)
        clf.fit(X, y)
        y_pred = clf.predict_proba(X)
        np.testing.assert_array_equal(first_pred, y_pred)


# Check simple cases.
def test_predict_proba():
    clf = TimeSeriesForestClassifier(n_estimators=2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], n_classes)
    np.testing.assert_array_equal(np.ones(n), np.sum(proba, axis=1))


# Compare results from different but equivalent implementations.
def test_different_implementations():
    # Due to tie-breaking/floating point rounding in the final decision tree classifier, the results depend on the
    # exact column order of the input data

    #  Compare pipeline predictions outside of ensemble.
    def _test_pipeline_predictions(n_intervals=None, random_state=None):
        steps = [
            ('segment', RandomIntervalSegmenter(n_intervals=n_intervals, check_input=False)),
            ('transform', TSFeatureUnion([
                ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
                ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False)))
            ])),
            ('clf', DecisionTreeClassifier())
        ]
        clf1 = TSPipeline(steps, random_state=random_state)
        clf1.fit(X_train, y_train)
        a = clf1.predict(X_test)

        steps = [
            ('transform', RandomIntervalFeatureExtractor(n_intervals=n_intervals, features=[np.mean, np.std])),
            ('clf', DecisionTreeClassifier())
        ]
        clf2 = TSPipeline(steps, random_state=random_state)
        clf2.fit(X_train, y_train)
        b = clf2.predict(X_test)
        np.array_equal(a, b)

    # Compare TimeSeriesForest ensemble predictions using pipeline as base_estimator
    def _test_TimeSeriesForest_predictions(n_estimators=None, n_intervals=None, random_state=None):

        # fully modular implementation using pipeline with FeatureUnion
        steps = [
            ('segment', RandomIntervalSegmenter(n_intervals=n_intervals, check_input=False)),
            ('transform', TSFeatureUnion([
                ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
                ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))),
                ('slope', RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False)))
            ])),
            ('clf', DecisionTreeClassifier())
        ]

        base_estimator = TSPipeline(steps)
        clf1 = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                          random_state=random_state,
                                          n_estimators=n_estimators)
        clf1.fit(X_train, y_train)
        a = clf1.predict_proba(X_test)

        # default, semi-modular implementation using RandomIntervalFeatureExtractor internally
        clf2 = TimeSeriesForestClassifier(random_state=random_state,
                                          n_estimators=n_estimators)
        clf2.set_params(**{'base_estimator__transform__n_intervals': n_intervals})
        clf2.fit(X_train, y_train)
        b = clf2.predict_proba(X_test)

        np.testing.assert_array_equal(a, b)

    X_train, y_train = load_gunpoint(return_X_y=True)
    X_test, y_test = load_gunpoint("TEST", return_X_y=True)
    random_state = 1234
    for n_intervals in ['sqrt', 'random', 1, 3]:
        _test_pipeline_predictions(n_intervals=n_intervals, random_state=random_state)

    for n_intervals in ['sqrt', 'random', 1, 3]:
        for n_estimators in [1, 3]:
            _test_TimeSeriesForest_predictions(n_estimators=n_estimators,
                                               n_intervals=n_intervals,
                                               random_state=random_state)

