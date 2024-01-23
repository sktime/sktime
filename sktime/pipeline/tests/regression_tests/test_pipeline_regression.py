import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.classification.dummy import DummyClassifier
from sktime.datasets import load_arrow_head, load_longley
from sktime.forecasting.compose import ForecastX
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.pipeline.pipeline import Pipeline
from sktime.split import temporal_train_test_split
from sktime.transformations.compose import Id
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.detrend import Detrender
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
    general_pipeline = Pipeline(
        [
            {
                "skobject": ExponentTransformer(),
                "name": "exponent",
                "edges": {"X": "X"},
            },
            {
                "skobject": BoxCoxTransformer(),
                "name": "BoxCOX",
                "edges": {"X": "exponent"},
            },
        ]
    )

    general_pipeline.fit(X=X)
    result_general = general_pipeline.transform(X)
    pd.testing.assert_frame_equal(result, result_general)


def test_classifier_regression():
    np.random.seed(42)
    X, y = load_arrow_head(split="train", return_X_y=True)
    clf_pipe = ExponentTransformer() * DummyClassifier()
    clf_pipe.fit(X, y)
    result = clf_pipe.predict(X)

    general_pipeline = Pipeline(
        [
            {
                "skobject": ExponentTransformer(),
                "name": "exponent",
                "edges": {"X": "X"},
            },
            {
                "skobject": DummyClassifier(),
                "name": "classifier",
                "edges": {"X": "exponent", "y": "y"},
            },
        ]
    )
    general_pipeline.fit(X=X, y=y)
    result_general = general_pipeline.predict(X)
    np.testing.assert_array_equal(result, result_general)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "method",
    [
        "predict",
        "predict_interval",
        "predict_quantiles",
        "predict_residuals",
    ],
)
def test_forecaster_regression(method):
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = getattr(pipe, method)(X=X_test)
    differencer = Differencer()

    general_pipeline = Pipeline(
        [
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
        ]
    )
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = getattr(general_pipeline, method)(X=X_test)
    np.testing.assert_array_equal(result, result_general)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_exogenous_transform_regression():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = ExponentTransformer() ** SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict(X=X_test)
    result_pi = pipe.predict_interval(X=X_test)

    general_pipeline = Pipeline(
        [
            {
                "skobject": ExponentTransformer(),
                "name": "exponent",
                "edges": {"X": "X"},
            },
            {
                "skobject": SARIMAX(),
                "name": "SARIMAX",
                "edges": {"X": "exponent", "y": "y"},
            },
        ]
    )

    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = general_pipeline.predict(X=X_test)
    result_pi_general = general_pipeline.predict_interval(X=X_test)
    np.testing.assert_array_equal(result, result_general)
    np.testing.assert_array_equal(result_pi, result_pi_general)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_endogenous_exogenous_transform_regression():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * ExponentTransformer() ** SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict(X=X_test)
    result_pi = pipe.predict_interval(X=X_test)
    differencer = Differencer()

    general_pipeline = Pipeline(
        [
            {"skobject": differencer, "name": "differencer", "edges": {"X": "y"}},
            {
                "skobject": ExponentTransformer(),
                "name": "exponent",
                "edges": {"X": "X"},
            },
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
        ]
    )
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = general_pipeline.predict(X=X_test)
    result_pi_general = general_pipeline.predict_interval(X=X_test)
    np.testing.assert_array_equal(result, result_general)
    np.testing.assert_array_equal(result_pi, result_pi_general)


def test_feature_union_regression():
    X = _bottom_hier_datagen(no_levels=1, no_bottom_nodes=2)
    pipe = Id() + Differencer() + Lag([1, 2], index_out="original")
    result = pipe.fit_transform(X)

    general_pipeline = Pipeline(
        [
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
        ]
    )

    result_general = general_pipeline.fit_transform(X=X)
    np.testing.assert_array_equal(result, result_general)


def test_feature_union_subsetting_regression():
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), n_columns=2, min_timepoints=3, max_timepoints=3
    )
    pipe = Id() + Differencer()["c0"] + Lag([1, 2], index_out="original")[["c1", "c0"]]
    result = pipe.fit_transform(X)

    general_pipeline = Pipeline(
        [
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
        ]
    )

    result_general = general_pipeline.fit_transform(X=X)
    np.testing.assert_array_equal(result, result_general)


@pytest.mark.parametrize(
    "data,testing_method",
    [
        (np.random.normal(0, 1, 200), np.testing.assert_array_equal),
        (
            pd.DataFrame(
                np.random.normal(0, 1, (200)),
                index=pd.DatetimeIndex(
                    pd.date_range("01.01.2008", freq="h", periods=200)
                ),
            ),
            pd.testing.assert_frame_equal,
        ),
        (
            pd.Series(
                np.random.normal(0, 1, 200),
                index=pd.DatetimeIndex(
                    pd.date_range("01.01.2008", freq="h", periods=200)
                ),
            ),
            pd.testing.assert_series_equal,
        ),
    ],
)
def test_varying_mtypes(data, testing_method):
    pipe = Detrender() * ExponentTransformer()
    pipe.fit(X=data)
    result = pipe.transform(X=data)

    general_pipeline = Pipeline(
        [
            {"skobject": Detrender(), "name": "detrender", "edges": {"X": "X"}},
            {
                "skobject": ExponentTransformer(),
                "name": "ExponentTransformer",
                "edges": {"X": "detrender"},
            },
        ]
    )

    general_pipeline.fit(X=data)
    general_pipeline.transform(X=data)
    result_general = general_pipeline.predict(data)
    testing_method(result, result_general)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_forecasterX_regression():
    y, X = load_longley()
    pipe = ForecastX(
        forecaster_X=NaiveForecaster(),
        forecaster_y=SARIMAX(),
    )
    pipe.fit(y, X=X, fh=[1, 2, 3])
    result = pipe.predict()

    general_pipeline = Pipeline(
        [
            {"skobject": NaiveForecaster(), "name": "forecastX", "edges": {"y": "X"}},
            {
                "skobject": SARIMAX(),
                "name": "forecastY",
                "edges": {"X": "forecastX", "y": "y"},
            },
        ]
    )

    general_pipeline.fit(y=y, X=X, fh=[1, 2, 3])
    result_general = general_pipeline.predict(None, None)
    pd.testing.assert_series_equal(result, result_general)
