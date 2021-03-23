# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sktime.series_as_features.compose import FeatureUnion
from sktime.transformations.panel.compose import (
    SeriesToPrimitivesRowTransformer,
)
from sktime.transformations.panel.segment import RandomIntervalSegmenter
from sktime.transformations.panel.summarize import (
    RandomIntervalFeatureExtractor,
)
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils._testing.panel import _make_nested_from_array
from sktime.utils.slope_and_trend import _slope


# Test output format and dimensions.
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [1, 3, "log", "sqrt", "random"])
@pytest.mark.parametrize(
    "features", [[np.mean], [np.mean, np.median], [np.mean, np.median, np.mean]]
)
def test_output_format_dim(n_instances, n_timepoints, n_intervals, features):
    X = _make_nested_from_array(
        np.ones(n_timepoints), n_instances=n_instances, n_columns=1
    )
    n_rows, n_cols = X.shape
    trans = RandomIntervalFeatureExtractor(n_intervals=n_intervals, features=features)
    Xt = trans.fit_transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == n_rows
    assert np.array_equal(Xt.values, np.ones(Xt.shape))


# Check that exception is raised for bad input args.
@pytest.mark.parametrize("bad_n_intervals", [0, "abc", 1.1, -1])
def test_bad_n_intervals(bad_n_intervals):
    X, y = make_classification_problem()
    with pytest.raises(ValueError):
        RandomIntervalFeatureExtractor(n_intervals=bad_n_intervals).fit(X)


@pytest.mark.parametrize(
    "bad_features", [0, "abc", {"a": 1}, (np.median, np.mean), [0, "abc"]]
)
def test_bad_features(bad_features):
    X, y = make_classification_problem()
    with pytest.raises(ValueError):
        RandomIntervalFeatureExtractor(n_intervals=bad_features).fit(X)


# Check specific results
@pytest.mark.parametrize("n_instances", [3, 5])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [1, 3, "log", "sqrt", "random"])
def test_results(n_instances, n_timepoints, n_intervals):
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


# Test against equivalent pipelines.
def test_different_implementations():
    random_state = 1233
    X_train, y_train = make_classification_problem()

    # Compare with chained transformations.
    tran1 = RandomIntervalSegmenter(n_intervals=1, random_state=random_state)
    tran2 = SeriesToPrimitivesRowTransformer(
        FunctionTransformer(func=np.mean, validate=False), check_transformer=False
    )
    A = tran2.fit_transform(tran1.fit_transform(X_train))

    tran = RandomIntervalFeatureExtractor(
        n_intervals=1, features=[np.mean], random_state=random_state
    )
    B = tran.fit_transform(X_train)

    np.testing.assert_array_almost_equal(A, B)


# Compare with transformer pipeline using TSFeatureUnion.
def test_different_pipelines():
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
                        SeriesToPrimitivesRowTransformer(
                            FunctionTransformer(func=np.mean, validate=False),
                            check_transformer=False,
                        ),
                    ),
                    (
                        "std",
                        SeriesToPrimitivesRowTransformer(
                            FunctionTransformer(func=np.std, validate=False),
                            check_transformer=False,
                        ),
                    ),
                    (
                        "slope",
                        SeriesToPrimitivesRowTransformer(
                            FunctionTransformer(func=_slope, validate=False),
                            check_transformer=False,
                        ),
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
