from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sktime.utils.time_series import time_series_slope
import pytest
import pandas as pd
import numpy as np


N_ITER = 10


# Test output format and dimensions.
def generate_df_from_array(array, n_rows=10, n_cols=1):
    return pd.DataFrame([[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


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

            multiplier_factor = 3
            Xt = trans.fit_transform(X * multiplier_factor)
            assert isinstance(Xt, pd.DataFrame)
            assert Xt.shape[0] == n_rows
            assert np.array_equal(Xt.values, np.ones(Xt.shape) * multiplier_factor)


def test_output_format_dim():
    for n_cols in range(1, 3, 10):
        for n_rows in [1, 3, 10]:
            for n_obs in [2, 3, 10]:
                X = generate_df_from_array(np.ones(n_obs), n_rows=n_rows, n_cols=n_cols)
                _test_output_format_dim(X)


# Check that exception is raised for bad input args.
def test_bad_input_args():
    bad_n_intervals = [0, 'abc', 1.0]
    for arg in bad_n_intervals:
        with pytest.raises(ValueError):
            RandomIntervalFeatureExtractor(n_intervals=arg)

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

