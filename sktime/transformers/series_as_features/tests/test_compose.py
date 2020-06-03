import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_basic_motions
from sktime.datasets import load_gunpoint
from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.compose import ColumnTransformer
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.utils.data_container import tabularize
from sktime.utils._testing import generate_df_from_array


def test_row_transformer_function_transformer_series_to_primitives():
    X, y = load_gunpoint(return_X_y=True)
    ft = FunctionTransformer(func=np.mean, validate=False)
    t = RowTransformer(ft)
    Xt = t.fit_transform(X, y)
    assert Xt.shape == X.shape
    assert isinstance(Xt.iloc[0, 0],
                      float)  # check series-to-primitive transforms


def test_row_transformer_function_transformer_series_to_series():
    X, y = load_gunpoint(return_X_y=True)

    # series-to-series transform function
    def powerspectrum(x):
        fft = np.fft.fft(x)
        ps = fft.real * fft.real + fft.imag * fft.imag
        return ps[:ps.shape[0] // 2]

    ft = FunctionTransformer(func=powerspectrum, validate=False)
    t = RowTransformer(ft)
    Xt = t.fit_transform(X, y)
    assert Xt.shape == X.shape
    assert isinstance(Xt.iloc[0, 0], (
        pd.Series, np.ndarray))  # check series-to-series transforms


def test_row_transformer_sklearn_transfomer():
    mu = 10
    X = generate_df_from_array(np.random.normal(loc=mu, scale=5, size=(100,)),
                               n_rows=10, n_cols=1)
    t = StandardScaler(with_mean=True, with_std=True)
    r = RowTransformer(t)

    Xt = r.fit_transform(X)
    assert Xt.shape == X.shape
    assert isinstance(Xt.iloc[0, 0], (
        pd.Series, np.ndarray))  # check series-to-series transform
    np.testing.assert_almost_equal(Xt.iloc[0, 0].mean(),
                                   0)  # check standardisation
    np.testing.assert_almost_equal(Xt.iloc[0, 0].std(), 1, decimal=2)


def test_row_transformer_transform_inverse_transform():
    X, y = load_gunpoint(return_X_y=True)
    t = RowTransformer(StandardScaler())
    Xt = t.fit_transform(X)
    Xit = t.inverse_transform(Xt)
    assert Xit.shape == X.shape
    assert isinstance(Xit.iloc[0, 0], (
        pd.Series, np.ndarray))  # check series-to-series transforms
    np.testing.assert_array_almost_equal(tabularize(X).values,
                                         tabularize(Xit).values, decimal=5)


def test_ColumnTransformer_pipeline():
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)

    # using Identity function transformers (transform series to series)
    def id_func(X):
        return X
    column_transformer = ColumnTransformer([
        ('id0', FunctionTransformer(func=id_func, validate=False), ['dim_0']),
        ('id1', FunctionTransformer(func=id_func, validate=False), ['dim_1'])
    ])
    steps = [
        ('extract', column_transformer),
        ('tabularise', Tabularizer()),
        ('classify', RandomForestClassifier(n_estimators=2, random_state=1))]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))


def test_RowTransformer_pipeline():
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)

    # using pure sklearn
    def row_mean(X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        Xt = pd.concat([pd.Series(col.apply(np.mean))
                        for _, col in X.items()], axis=1)
        return Xt

    def row_first(X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        Xt = pd.concat([pd.Series(tabularize(col).iloc[:, 0])
                        for _, col in X.items()], axis=1)
        return Xt

    # specify column as a list, otherwise pandas Series are selected and
    # passed on to the transformers
    transformer = ColumnTransformer([
        ('mean', FunctionTransformer(func=row_mean, validate=False),
         ['dim_0']),
        ('first', FunctionTransformer(func=row_first, validate=False),
         ['dim_1'])
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
        ('mean',
         RowTransformer(FunctionTransformer(func=np.mean, validate=False)),
         ['dim_0']),
        ('first', FunctionTransformer(func=row_first, validate=False),
         ['dim_1'])
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
