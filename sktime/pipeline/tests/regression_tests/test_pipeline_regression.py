import numpy as np
import pandas as pd

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_longley, load_arrow_head
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.sarimax import SARIMAX
from sktime.pipeline.pipeline import Pipeline
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.exponent import ExponentTransformer


def test_transformer_regression():
    np.random.seed(42)
    y, X = load_longley()
    trafo_pipe = ExponentTransformer() * BoxCoxTransformer()
    trafo_pipe.fit(X=X)
    result = trafo_pipe.transform(X)
    np.random.seed(42)
    general_pipeline = Pipeline()
    for step in [{"skobject" : ExponentTransformer(),
                  "name":"exponent",
                  "edges": {"X":"X"}},
                 {"skobject" : BoxCoxTransformer(),
                  "name":"BoxCOX",
                  "edges": {"X":"exponent"}}]:
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
    for step in [{"skobject" : ExponentTransformer(),
                  "name":"exponent",
                  "edges": {"X":"X"}},
                 {"skobject" : KNeighborsTimeSeriesClassifier(),
                  "name":"BoxCOX",
                  "edges": {"X":"exponent", "y":"y"}}]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(X=X, y=y)
    result_general = general_pipeline.predict(X)
    np.testing.assert_array_equal(result, result_general)


def test_forecaster_regression():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict(X=X_test)
    result_pi = pipe.predict_interval(X=X_test)
    general_pipeline = Pipeline()
    differencer = Differencer()
    for step in [{"skobject" : differencer,
                  "name":"differencer",
                  "edges": {"X":"y"}},
                 {"skobject" : SARIMAX(),
                  "name":"SARIMAX",
                  "edges": {"X":"X", "y":"differencer"}},
                 {"skobject" : differencer,
                  "name":"differencer_inverse",
                  "edges": {"X":"SARIMAX"},
                  "method": "inverse_transform",
                  }
                 ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = general_pipeline.predict(X=X_test)
    result_pi_general = general_pipeline.predict_interval(X=X_test)
    np.testing.assert_array_equal(result, result_general)
    np.testing.assert_array_equal(result_pi, result_pi_general)

def test_exogenous_transform_regression():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = ExponentTransformer() ** SARIMAX()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict(X=X_test)
    result_pi = pipe.predict_interval(X=X_test)
    general_pipeline = Pipeline()
    for step in [{"skobject" : ExponentTransformer(),
                  "name":"exponent",
                  "edges": {"X":"X"}},
                 {"skobject" : SARIMAX(),
                  "name":"SARIMAX",
                  "edges": {"X":"exponent", "y":"y"}}
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
    for step in [{"skobject" : differencer,
                  "name":"differencer",
                  "edges": {"X":"y"}},
                 {"skobject" : ExponentTransformer(),
                  "name":"exponent",
                  "edges": {"X":"X"}},
                 {"skobject" : SARIMAX(),
                  "name":"SARIMAX",
                  "edges": {"X":"exponent", "y":"differencer"}},
                 {"skobject" : differencer,
                  "name":"differencer_inverse",
                  "edges": {"X":"SARIMAX"},
                  "method": "inverse_transform",
                  }
                 ]:
        general_pipeline.add_step(**step)
    general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result_general = general_pipeline.predict(X=X_test)
    result_pi_general = general_pipeline.predict_interval(X=X_test)
    np.testing.assert_array_equal(result, result_general)
    np.testing.assert_array_equal(result_pi, result_pi_general)
