from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sktime.utils.testing import generate_df_from_array
import pytest
import pandas as pd
import numpy as np
from sktime.transformers.compose import RowwiseTransformer
from sktime.datasets import load_gunpoint
from sktime.transformers.series_to_series import RandomIntervalSegmenter
from sklearn.preprocessing import FunctionTransformer
from sktime.utils.time_series import time_series_slope
from sktime.pipeline import Pipeline, FeatureUnion

N_ITER = 10


# Test output format and dimensions.
def _test_output_format_dim(X):
    n_rows, n_cols = X.shape
    n_intervals_args = [1, 3, 10, 'sqrt', 'random']
    feature_args = [[np.mean], [np.mean, np.median], [np.mean, np.median, np.mean]]
    for n_intervals in n_intervals_args:
        for features in feature_args:
            trans = RandomIntervalFeatureExtractor(n_intervals=n_intervals, features=features)
            Xt = trans.fit_transform(X)
            assert isinstance(Xt, pd.DataFrame)
            assert Xt.shape[0] == n_rows
            assert np.array_equal(Xt.values, np.ones(Xt.shape))


def test_output_format_dim():
    for n_cols in range(1, 3, 10):
        for n_rows in [1, 3, 10]:
            for n_obs in [2, 3, 10]:
                X = generate_df_from_array(np.ones(n_obs), n_rows=n_rows, n_cols=n_cols)
                _test_output_format_dim(X)


# Check that exception is raised for bad input args.
def test_bad_input_args():
    bad_features = [0, 'abc', {'a': 1}, (np.median, np.mean), [0, 'abc']]
    for arg in bad_features:
        with pytest.raises(ValueError):
            RandomIntervalFeatureExtractor(features=arg)


# Check if random state always gives same results
def test_random_state():
    X = generate_df_from_array(np.random.normal(size=20))
    random_state = 1234
    trans = RandomIntervalFeatureExtractor(n_intervals='random', random_state=random_state)
    first_Xt = trans.fit_transform(X)
    for _ in range(N_ITER):
        trans = RandomIntervalFeatureExtractor(n_intervals='random', random_state=random_state)
        Xt = trans.fit_transform(X)
        assert first_Xt.equals(Xt)


# Check specific results
def _test_results(n_cols, n_obs, n_intervals):
    x = np.random.normal(size=n_obs)
    X = generate_df_from_array(x, n_rows=3, n_cols=n_cols)
    trans = RandomIntervalFeatureExtractor(n_intervals=n_intervals,
                                           features=[np.mean, np.std, time_series_slope])
    Xt = trans.fit_transform(X)

    # Check results
    for c in range(n_cols):
        for s, e in trans.intervals_[c]:
            assert np.all(Xt.filter(like=f'_{s}_{e}_mean') == np.mean(x[s:e]))
            assert np.all(Xt.filter(like=f'_{s}_{e}_std') == np.std(x[s:e]))
            assert np.all(Xt.filter(like=f'_{s}_{e}_time_series_slope') == time_series_slope(x[s:e]))


def test_results():
    for n_cols in [1, 3]:
        for n_obs in [2, 10]:
            for n_intervals in [1, 3, 10]:
                _test_results(n_cols, n_obs, n_intervals)


# Test against equivalent pipelines.
def test_different_implementations():
    random_seed = 1233
    X_train, y_train = load_gunpoint(return_X_y=True)

    # Compare with chained transformations.
    tran1 = RandomIntervalSegmenter(n_intervals='sqrt', random_state=random_seed)
    tran2 = RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))
    a = tran2.fit_transform(tran1.fit_transform(X_train))

    tran = RandomIntervalFeatureExtractor(n_intervals='sqrt', features=[np.mean], random_state=random_seed)
    b = tran.fit_transform(X_train)

    np.testing.assert_array_equal(a, b)

    # Compare with transformer pipeline using TSFeatureUnion.
    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals='sqrt', check_input=False)),
        ('transform', FeatureUnion([
            ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
            ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))),
        ])),
    ]
    pipe = Pipeline(steps, random_state=random_seed)
    a = pipe.fit_transform(X_train)
    n_ints = a.shape[1] // 2  # Rename columns for comparing re-ordered arrays.
    a.columns = [*a.columns[:n_ints] + '_mean', *a.columns[n_ints:n_ints * 2] + '_std']
    a = a.reindex(np.sort(a.columns), axis=1)

    tran = RandomIntervalFeatureExtractor(n_intervals='sqrt', features=[np.mean, np.std],
                                          random_state=random_seed)
    b = tran.fit_transform(X_train)
    b = b.reindex(np.sort(b.columns), axis=1)
    np.testing.assert_array_equal(a, b)
