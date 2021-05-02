# -*- coding: utf-8 -*-
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.pipeline.pipeline import NetworkPipeline
from sktime.transformations.series.impute import Imputer

if __name__ == "__main__":
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
                {"fit": {"y": "original_y"}, "predict": {"fh": "original_y"}},
            ),
        ]
    )
    pipe.fit(y=y_train)
    pipe.predict(y=fh)
