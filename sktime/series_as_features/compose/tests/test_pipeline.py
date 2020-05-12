import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from sktime.datasets import load_gunpoint
from sktime.series_as_features.compose.pipeline import FeatureUnion
from sktime.series_as_features.compose.pipeline import Pipeline
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.segment import \
    RandomIntervalSegmenter
from sktime.transformers.series_as_features.summarize import \
    RandomIntervalFeatureExtractor

# load data
X_train, y_train = load_gunpoint("TRAIN", return_X_y=True)
X_train = pd.concat([X_train, X_train], axis=1)
X_train.columns = ['ts', 'ts_copy']

X_test, y_test = load_gunpoint("TEST", return_X_y=True)
X_test = pd.concat([X_test, X_test], axis=1)
X_test.columns = ['ts', 'ts_copy']


def test_FeatureUnion_pipeline():
    # pipeline with segmentation plus multiple feature extraction
    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals=3)),
        ('transform', FeatureUnion([
            ('mean', RowTransformer(
                FunctionTransformer(func=np.mean, validate=False))),
            ('std',
             RowTransformer(FunctionTransformer(func=np.std, validate=False)))
        ])),
        ('clf', DecisionTreeClassifier())
    ]
    clf = Pipeline(steps)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))


def test_Pipeline_random_state():
    steps = [('transform', RandomIntervalFeatureExtractor(features=[np.mean])),
             ('clf', DecisionTreeClassifier())]
    pipe = Pipeline(steps)

    # Check that pipe is initiated without random_state
    assert pipe.random_state is None
    assert pipe.get_params()['random_state'] is None

    # Check that all components are initiated without random_state
    for step in pipe.steps:
        assert step[1].random_state is None
        assert step[1].get_params()['random_state'] is None

    # Check that if random state is set, it's set to itself and all its
    # random components
    rs = 1234
    pipe.set_params(**{'random_state': rs})

    assert pipe.random_state == rs
    assert pipe.get_params()['random_state'] == rs

    for step in pipe.steps:
        assert step[1].random_state == rs
        assert step[1].get_params()['random_state'] == rs

    # Check specific results
    X_train, y_train = load_gunpoint(return_X_y=True)
    X_test, y_test = load_gunpoint("TEST", return_X_y=True)

    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals=3)),
        ('extract',
         RowTransformer(FunctionTransformer(func=np.mean, validate=False))),
        ('clf', DecisionTreeClassifier())
    ]
    pipe = Pipeline(steps, random_state=rs)
    pipe.fit(X_train, y_train)
    y_pred_first = pipe.predict(X_test)
    N_ITER = 10
    for _ in range(N_ITER):
        pipe = Pipeline(steps, random_state=rs)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        np.testing.assert_array_equal(y_pred_first, y_pred)


def test_FeatureUnion():
    X, y = load_gunpoint(return_X_y=True)
    ft = FunctionTransformer(func=np.mean, validate=False)
    t = RowTransformer(ft)
    fu = FeatureUnion([
        ('mean', t),
        ('std',
         RowTransformer(FunctionTransformer(func=np.std, validate=False)))
    ])
    Xt = fu.fit_transform(X, y)
    assert Xt.shape == (X.shape[0], X.shape[1] * len(fu.transformer_list))
