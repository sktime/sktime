import numpy as np
import pandas as pd
import pytest

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_arrow_head, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.sarimax import SARIMAX
from sktime.pipeline.pipeline import Pipeline
from sktime.transformations.compose import Id
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.lag import Lag
from sktime.utils._testing.hierarchical import _bottom_hier_datagen, _make_hierarchical


def test_transformer_regression():
    np.random.seed(42)
    y, X = load_longley()
    trafo_pipe = ExponentTransformer() * BoxCoxTransformer()
    trafo_pipe.fit(X=X)
    result = trafo_pipe.transform(X)

    np.random.seed(42)
    general_pipeline = Pipeline()
    for step in [
        {"skobject": ExponentTransformer(), "name": "exponent", "edges": {"X": "X"}},
        {"skobject": BoxCoxTransformer(), "name": "BoxCOX", "edges": {"X": "exponent"}},
    ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(X=X)
    result_general = general_pipeline.transform(X)
    pd.testing.assert_frame_equal(result, result_general)


def test_classifier_regression():
    np.random.seed(42)
    X, y = load_arrow_head(split="train", return_X_y=True)
    clf_pipe = ExponentTransformer() * KNeighborsTimeSeriesClassifier()
    clf_pipe.fit(X, y)
    result = clf_pipe.predict(X)

    general_pipeline = Pipeline()
    for step in [
        {"skobject": ExponentTransformer(), "name": "exponent", "edges": {"X": "X"}},
        {
            "skobject": KNeighborsTimeSeriesClassifier(),
            "name": "BoxCOX",
            "edges": {"X": "exponent", "y": "y"},
        },
    ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(X=X, y=y)
    result_general = general_pipeline.predict(X)
    np.testing.assert_array_equal(result, result_general)


@pytest.mark.parametrize(
    "method",
    [
        "predict",
        "predict_interval",
        "predict_quantiles",
    ],
)
def test_forecaster_regression(method):
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = getattr(pipe, method)(X=X_test)

    general_pipeline = Pipeline()
    differencer = Differencer()
    for step in [
        {"skobject": differencer, "name": "differencer", "edges": {"X": "y"}},
        {
            "skobject": SARIMAX(),
            "name": "SARIMAX",
            "edges": {"X": "X", "y": "differencer"},
        },
        {
            "skobject": differencer,
            "name": "differencer_inverse",
            "edges": {"X": "SARIMAX"},
            "method": "inverse_transform",
        },
    ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = getattr(general_pipeline, method)(X=X_test)
    np.testing.assert_array_equal(result, result_general)


def test_forecaster_regression_predict_residuals():
    # TODO integrate in test_forecaster_regression if issue 4766 is fixed
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * SARIMAX()
    pipe.fit(y=y_train.to_frame(), X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict_residuals()

    general_pipeline = Pipeline()
    differencer = Differencer()
    for step in [
        {"skobject": differencer, "name": "differencer", "edges": {"X": "y"}},
        {
            "skobject": SARIMAX(),
            "name": "SARIMAX",
            "edges": {"X": "X", "y": "differencer"},
        },
        {
            "skobject": differencer,
            "name": "differencer_inverse",
            "edges": {"X": "SARIMAX"},
            "method": "inverse_transform",
        },
    ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = pipe.predict_residuals()
    np.testing.assert_array_equal(result, result_general)


def test_exogenous_transform_regression():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = ExponentTransformer() ** SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict(X=X_test)
    result_pi = pipe.predict_interval(X=X_test)

    general_pipeline = Pipeline()
    for step in [
        {"skobject": ExponentTransformer(), "name": "exponent", "edges": {"X": "X"}},
        {
            "skobject": SARIMAX(),
            "name": "SARIMAX",
            "edges": {"X": "exponent", "y": "y"},
        },
    ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = general_pipeline.predict(X=X_test)
    result_pi_general = general_pipeline.predict_interval(X=X_test)
    np.testing.assert_array_equal(result, result_general)
    np.testing.assert_array_equal(result_pi, result_pi_general)


def test_endogenous_exogenous_transform_regression():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * ExponentTransformer() ** SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict(X=X_test)
    result_pi = pipe.predict_interval(X=X_test)

    general_pipeline = Pipeline()
    differencer = Differencer()
    for step in [
        {"skobject": differencer, "name": "differencer", "edges": {"X": "y"}},
        {"skobject": ExponentTransformer(), "name": "exponent", "edges": {"X": "X"}},
        {
            "skobject": SARIMAX(),
            "name": "SARIMAX",
            "edges": {"X": "exponent", "y": "differencer"},
        },
        {
            "skobject": differencer,
            "name": "differencer_inverse",
            "edges": {"X": "SARIMAX"},
            "method": "inverse_transform",
        },
    ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = general_pipeline.predict(X=X_test)
    result_pi_general = general_pipeline.predict_interval(X=X_test)
    np.testing.assert_array_equal(result, result_general)
    np.testing.assert_array_equal(result_pi, result_pi_general)


def test_feature_union_regression():
    X = _bottom_hier_datagen(no_levels=1, no_bottom_nodes=2)
    pipe = Id() + Differencer() + Lag([1, 2], index_out="original")
    result = pipe.fit_transform(X)

    general_pipeline = Pipeline()
    for step in [
        {"skobject": Id(), "name": "id", "edges": {"X": "X"}},
        {"skobject": Differencer(), "name": "differencer", "edges": {"X": "X"}},
        {
            "skobject": Lag([1, 2], index_out="original"),
            "name": "lag",
            "edges": {"X": "X"},
        },
        {
            "skobject": Id(),
            "name": "combined",
            "edges": {"X": ["id", "differencer", "lag"]},
        },
    ]:
        general_pipeline.add_step(**step)
    result_general = general_pipeline.fit_transform(X=X)
    np.testing.assert_array_equal(result, result_general)


def test_feature_union_subsetting_regression():
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), n_columns=2, min_timepoints=3, max_timepoints=3
    )
    pipe = Id() + Differencer()["c0"] + Lag([1, 2], index_out="original")[["c1", "c0"]]
    result = pipe.fit_transform(X)

    general_pipeline = Pipeline()
    for step in [
        {"skobject": Id(), "name": "id", "edges": {"X": "X"}},
        {"skobject": Differencer(), "name": "differencer", "edges": {"X": "X__c0"}},
        {
            "skobject": Lag([1, 2], index_out="original"),
            "name": "lag",
            "edges": {"X": "X__c1_c0"},
        },
        {
            "skobject": Id(),
            "name": "combined",
            "edges": {"X": ["id", "differencer", "lag"]},
        },
    ]:
        general_pipeline.add_step(**step)
    result_general = general_pipeline.fit_transform(X=X)
    np.testing.assert_array_equal(result, result_general)
