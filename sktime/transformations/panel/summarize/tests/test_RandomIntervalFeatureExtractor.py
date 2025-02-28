"""Tests for RIFE."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import FeatureUnion
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.segment import RandomIntervalSegmenter
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils._testing.panel import (
    _make_nested_from_array,
    make_classification_problem,
)
from sktime.utils.slope_and_trend import _slope


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [1, 3, "log", "sqrt", "random"])
@pytest.mark.parametrize(
    "features", [[np.mean], [np.mean, np.median], [np.mean, np.median, np.mean]]
)
def test_output_format_dim(n_instances, n_timepoints, n_intervals, features):
    """Test output format and dimensions."""
    X = _make_nested_from_array(
        np.ones(n_timepoints), n_instances=n_instances, n_columns=1
    )
    n_rows, n_cols = X.shape
    trans = RandomIntervalFeatureExtractor(n_intervals=n_intervals, features=features)
    Xt = trans.fit_transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == n_rows
    assert np.array_equal(Xt.values, np.ones(Xt.shape))


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_n_intervals", [0, "abc", 1.1, -1])
def test_bad_n_intervals(bad_n_intervals):
    """Check that exception is raised for bad input args."""
    X, y = make_classification_problem()
    with pytest.raises(ValueError):
        RandomIntervalFeatureExtractor(n_intervals=bad_n_intervals).fit(X)


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "bad_features", [0, "abc", {"a": 1}, (np.median, np.mean), [0, "abc"]]
)
def test_bad_features(bad_features):
    """Test that ValueError is raised for bad features."""
    X, y = make_classification_problem()
    with pytest.raises(ValueError):
        RandomIntervalFeatureExtractor(n_intervals=bad_features).fit(X)


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_instances", [3, 5])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [1, 3, "log", "sqrt", "random"])
def test_results(n_instances, n_timepoints, n_intervals):
    """Check specific results."""
    X, _ = make_classification_problem(
        n_instances=n_instances, n_timepoints=n_timepoints, return_numpy=True
    )
    transformer = RandomIntervalFeatureExtractor(
        n_intervals=n_intervals, features=[np.mean, np.std]
    )
    Xt = transformer.fit_transform(X)
    Xt = Xt.loc[:, ~Xt.columns.duplicated()]
    # Check results
    intervals = transformer.intervals_
    for start, end in intervals:
        expected_mean = np.mean(X[:, 0, start:end], axis=-1)
        expected_std = np.std(X[:, 0, start:end], axis=-1)

        actual_means = Xt.loc[:, f"{start}_{end}_mean"].to_numpy().ravel()
        actual_stds = Xt.loc[:, f"{start}_{end}_std"].to_numpy().ravel()

        np.testing.assert_array_equal(actual_means, expected_mean)
        np.testing.assert_array_equal(actual_stds, expected_std)


@pytest.mark.skipif(
    not run_test_for_class([RandomIntervalFeatureExtractor, RandomIntervalSegmenter]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_different_implementations():
    """Test against equivalent pipelines."""
    random_state = 1233
    X_train, y_train = make_classification_problem()

    # Compare with chained transformations.
    tran1 = RandomIntervalSegmenter(n_intervals=1, random_state=random_state)
    tran2 = FunctionTransformer(func=np.mean, validate=False)
    t_chain = tran1 * tran2
    A = t_chain.fit_transform(X_train)

    tran = RandomIntervalFeatureExtractor(
        n_intervals=1, features=[np.mean], random_state=random_state
    )
    B = tran.fit_transform(X_train)

    np.testing.assert_array_almost_equal(A, B)


@pytest.mark.skipif(
    not run_test_for_class([RandomIntervalFeatureExtractor, RandomIntervalSegmenter]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.xfail(reason="SeriesToPrimitivesTransformer will be deprecated, see 2179")
def test_different_pipelines():
    """Compare with transformer pipeline using TSFeatureUnion."""
    random_state = 1233
    X_train, y_train = make_classification_problem()
    steps = [
        (
            "segment",
            RandomIntervalSegmenter(n_intervals=1, random_state=random_state),
        ),
        (
            "transform",
            FeatureUnion(
                [
                    (
                        "mean",
                        Tabularizer(FunctionTransformer(func=np.mean, validate=False)),
                    ),
                    (
                        "std",
                        Tabularizer(FunctionTransformer(func=np.std, validate=False)),
                    ),
                    (
                        "slope",
                        Tabularizer(FunctionTransformer(func=_slope, validate=False)),
                    ),
                ]
            ),
        ),
    ]
    pipe = Pipeline(steps)
    a = pipe.fit_transform(X_train)
    tran = RandomIntervalFeatureExtractor(
        n_intervals=1,
        features=[np.mean, np.std, _slope],
        random_state=random_state,
    )
    b = tran.fit_transform(X_train)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(pipe.steps[0][1].intervals_, tran.intervals_)
