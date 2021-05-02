# -*- coding: utf-8 -*-
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.pipeline.pipeline import OnlineUnsupervisedPipeline

if __name__ == "__main__":
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = NaiveForecaster(strategy="last", sp=12)
    pipe = OnlineUnsupervisedPipeline(
        steps=[
            (
                "forecaster",
                forecaster,
                {"fit": {"y": "original"}, "predict": {"fh": "original"}},
            )
        ]
    )
    pipe.fit(y_train)
    pipe.predict(fh)
