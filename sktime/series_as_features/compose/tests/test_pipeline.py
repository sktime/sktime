import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sktime.datasets import load_gunpoint
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.segment import \
    RandomIntervalSegmenter

# load data
X_train, y_train = load_gunpoint("train", return_X_y=True)
X_train = pd.concat([X_train, X_train], axis=1)
X_train.columns = ['ts', 'ts_copy']

X_test, y_test = load_gunpoint("test", return_X_y=True)
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
