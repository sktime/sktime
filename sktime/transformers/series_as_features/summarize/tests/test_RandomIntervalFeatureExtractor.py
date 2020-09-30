import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sktime.utils._testing import make_classification_problem
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.segment import \
    RandomIntervalSegmenter
from sktime.transformers.series_as_features.summarize import \
    RandomIntervalFeatureExtractor
from sktime.utils._testing import generate_df_from_array
from sktime.utils.time_series import time_series_slope


# Test output format and dimensions.
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [1, 3, 'log', 'sqrt', 'random'])
@pytest.mark.parametrize("features", [[np.mean], [np.mean, np.median],
                                      [np.mean, np.median, np.mean]])
def test_output_format_dim(n_instances, n_timepoints, n_intervals, features):
    X = generate_df_from_array(np.ones(n_timepoints), n_rows=n_instances,
                               n_cols=1)
    n_rows, n_cols = X.shape
    trans = RandomIntervalFeatureExtractor(n_intervals=n_intervals,
                                           features=features)
    Xt = trans.fit_transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == n_rows
    assert np.array_equal(Xt.values, np.ones(Xt.shape))


# Check that exception is raised for bad input args.
@pytest.mark.parametrize("bad_n_intervals", [0, 'abc', 1.1, -1])
def test_bad_n_intervals(bad_n_intervals):
    X, y = make_classification_problem()
    with pytest.raises(ValueError):
        RandomIntervalFeatureExtractor(n_intervals=bad_n_intervals).fit(X)


@pytest.mark.parametrize("bad_features",
                         [0, 'abc', {'a': 1}, (np.median, np.mean),
                          [0, 'abc']])
def test_bad_features(bad_features):
    X, y = make_classification_problem()
    with pytest.raises(ValueError):
        RandomIntervalFeatureExtractor(n_intervals=bad_features).fit(X)


# Check specific results
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [1, 3, 'log', 'sqrt', 'random'])
def test_results(n_instances, n_timepoints, n_intervals):
    x = np.random.normal(size=n_timepoints)
    X = generate_df_from_array(x, n_rows=n_instances, n_cols=1)
    t = RandomIntervalFeatureExtractor(n_intervals=n_intervals,
                                       features=[np.mean, np.std,
                                                 time_series_slope])
    Xt = t.fit_transform(X)
    # Check results
    intervals = t.intervals_
    for start, end in intervals:
        expected_mean = np.mean(x[start:end])
        expected_std = np.std(x[start:end])
        expected_slope = time_series_slope(x[start:end])

        actual_means = Xt.filter(like=f'*_{start}_{end}_mean').values
        actual_stds = Xt.filter(like=f'_{start}_{end}_std').values
        actual_slopes = Xt.filter(
            like=f'_{start}_{end}_time_series_slope').values

        assert np.all(actual_means == expected_mean)
        assert np.all(actual_stds == expected_std)
        assert np.all(actual_slopes == expected_slope)


# Test against equivalent pipelines.
def test_different_implementations():
    random_state = 1233
    X_train, y_train = make_classification_problem()

    # Compare with chained transformations.
    tran1 = RandomIntervalSegmenter(n_intervals='sqrt',
                                    random_state=random_state)
    tran2 = RowTransformer(FunctionTransformer(func=np.mean, validate=False))
    A = tran2.fit_transform(tran1.fit_transform(X_train))

    tran = RandomIntervalFeatureExtractor(n_intervals='sqrt',
                                          features=[np.mean],
                                          random_state=random_state)
    B = tran.fit_transform(X_train)

    np.testing.assert_array_equal(A, B)


# Compare with transformer pipeline using TSFeatureUnion.
def test_different_pipelines():
    random_state = 1233
    X_train, y_train = make_classification_problem()
    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals='sqrt',
                                            random_state=random_state)),
        ('transform', FeatureUnion([
            ('mean', RowTransformer(
                FunctionTransformer(func=np.mean, validate=False))),
            ('std',
             RowTransformer(FunctionTransformer(func=np.std, validate=False))),
            ('slope', RowTransformer(
                FunctionTransformer(func=time_series_slope, validate=False))),
        ])),
    ]
    pipe = Pipeline(steps)
    a = pipe.fit_transform(X_train)
    tran = RandomIntervalFeatureExtractor(n_intervals='sqrt',
                                          features=[np.mean, np.std,
                                                    time_series_slope],
                                          random_state=random_state)
    b = tran.fit_transform(X_train)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(pipe.steps[0][1].intervals_, tran.intervals_)
