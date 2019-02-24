from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
import pytest
import pandas as pd
import numpy as np


# Test output format and dimensions.
def generate_ones_df(n_rows=10, n_cols=1, n_obs=10):
    return pd.DataFrame([[pd.Series(np.ones(n_obs)) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


def generate_random_df(n_rows=10, n_cols=1, n_obs=10):
    return pd.DataFrame([[pd.Series(np.random.normal(size=n_obs)) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


def _test_output_format_dim(X):
    n_rows, n_cols = X.shape
    n_intervals_args = [1, 10, 'sqrt', 'random']
    feature_args = [[np.mean], [np.mean, np.median], [np.mean, np.median, np.mean]]
    for n_intervals in n_intervals_args:
        for features in feature_args:
            trans = RandomIntervalFeatureExtractor(n_intervals=n_intervals, features=features)
            Xt = trans.fit_transform(X)

            assert isinstance(Xt, pd.DataFrame)
            assert Xt.shape[0] == n_rows
            assert np.array_equal(Xt.values, np.ones(Xt.shape))


def test_output_format_dim():
    for n_cols in range(1, 3):
        for n_rows in [5, 10]:
            X = generate_ones_df(n_rows=n_rows, n_cols=n_cols)
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
    X = generate_random_df()
    random_state = 1234
    trans = RandomIntervalFeatureExtractor(n_intervals='random', random_state=random_state)
    first_Xt = trans.fit_transform(X)
    for _ in range(100):
        trans = RandomIntervalFeatureExtractor(n_intervals='random', random_state=random_state)
        Xt = trans.fit_transform(X)
        assert first_Xt.equals(Xt)
