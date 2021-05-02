# -*- coding: utf-8 -*-
from sktime.datasets import load_airline
from sktime.datasets import load_arrow_head

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.pipeline.pipeline import NetworkPipeline
from sktime.transformations.series.impute import Imputer

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sktime.transformations.base import BaseTransformer
from sktime.utils.data_processing import from_nested_to_2d_array


def test_airline():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = NaiveForecaster(strategy="last", sp=12)
    pipe = NetworkPipeline(
        steps=[
            (
                "imputer",
                Imputer(method="drift"),
                {"fit": {"Z": "original_y"}, "predict": None},
            ),
            (
                "forecaster",
                forecaster,
                {"fit": {"y": "imputer"}, "predict": {"fh": "original_y"}},
            ),
        ]
    )
    pipe.fit(y=y_train)
    pipe.predict(y=fh)


class Tabularizer(BaseTransformer):
    def fit_transform(self, X):
        return from_nested_to_2d_array(X)

    def transform(self, X):
        return self.fit_transform(X)


def test_arrowhead():
    X, y = load_arrow_head(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipe = NetworkPipeline(
        steps=[
            ("tabularizer", Tabularizer(), {"X": "original_X"}),
            (
                "classifier",
                DummyClassifier(strategy="prior"),
                {
                    "fit": {"X": "tabularizer", "y": "original_y"},
                    "predict": {"X": "tabularizer"},
                },
            ),
        ]
    )
    pipe.fit(X=X_train, y=y_train)
    pipe.predict(X=X_test)
