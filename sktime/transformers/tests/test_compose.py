import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

from sktime.pipeline import Pipeline
from sktime.tests.test_pipeline import X_train, y_train, X_test, y_test
from sktime.transformers.compose import ColumnTransformer, Tabulariser, RowwiseTransformer
from sktime.datasets import load_basic_motions
from sktime.utils.data_container import tabularise

# load data
X_train, y_train = load_basic_motions("TRAIN", return_X_y=True)
X_test, y_test = load_basic_motions("TEST", return_X_y=True)


def test_Rowwise_transformer():
    X, y = load_basic_motions(return_X_y=True)
    ft = FunctionTransformer(func=np.mean, validate=False)
    t = RowwiseTransformer(ft)
    Xt = t.fit_transform(X, y)
    assert Xt.shape == X.shape

def test_ColumnTransformer_pipeline():
    # using Identity function transformers (transform series to series)
    id_func = lambda X: X
    column_transformer = ColumnTransformer([
        ('id0', FunctionTransformer(func=id_func, validate=False), ['dim_0']),
        ('id1', FunctionTransformer(func=id_func, validate=False), ['dim_1'])
    ])
    steps = [
        ('extract', column_transformer),
        ('tabularise', Tabulariser()),
        ('classify', RandomForestClassifier(n_estimators=2))]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))


def test_RowwiseTransformer_pipeline():

    # using pure sklearn
    def rowwise_mean(X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        Xt = pd.concat([pd.Series(col.apply(np.mean))
                        for _, col in X.items()], axis=1)
        return Xt

    def rowwise_first(X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        Xt = pd.concat([pd.Series(tabularise(col).iloc[:, 0])
                        for _, col in X.items()], axis=1)
        return Xt

    # specify column as a list, otherwise pandas Series are selected and passed on to the transformers
    transformer = ColumnTransformer([
        ('mean', FunctionTransformer(func=rowwise_mean, validate=False), ['dim_0']),
        ('first', FunctionTransformer(func=rowwise_first, validate=False), ['dim_1'])
    ])
    estimator = RandomForestClassifier(n_estimators=2, random_state=1)
    steps = [
        ('extract', transformer),
        ('classify', estimator)
    ]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    expected = model.predict(X_test)

    # using sktime with sklearn pipeline
    transformer = ColumnTransformer([
        ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False)), ['dim_0']),
        ('first', FunctionTransformer(func=rowwise_first, validate=False), ['dim_1'])
    ])
    estimator = RandomForestClassifier(n_estimators=2, random_state=1)
    steps = [
        ('extract', transformer),
        ('classify', estimator)
    ]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    actual = model.predict(X_test)
    np.testing.assert_array_equal(expected, actual)
