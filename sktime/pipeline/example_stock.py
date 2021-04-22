# -*- coding: utf-8 -*-
import pandas_datareader as pdr
from sktime.pipeline.transformers import MovingAverage
from sktime.pipeline.transformers import RelativeStrengthIndex
from sktime.pipeline.transformers import DateFeatures
from sktime.pipeline.transformers import FixedTimeHorizonLabels
from sktime.pipeline.transformers import DatasetConcatenator
from sktime.pipeline.transformers import (
    is_friday,
    is_month_end,
    is_quarter_end,
    is_year_end,
)
from sktime.pipeline.pipeline import OnlineUnsupervisedPipeline

# from sktime.pipeline.estimators import RandomForestRegressorWrapper
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    aapl = pdr.DataReader(
        "AAPL", data_source="yahoo", start="2000-01-01", end="2021-03-31"
    )
    train_idx = int(aapl.shape[0] * 2 / 3)
    aapl_train = aapl.iloc[0:train_idx]
    aapl_test = aapl.iloc[train_idx + 1 : -1]
    # simple example
    pipe = OnlineUnsupervisedPipeline(
        steps=[
            (
                "ma_10",
                MovingAverage(input_col="Close"),
                {"input_series": "original", "window": 10},
            ),
            (
                "ma_100",
                MovingAverage(input_col="Close"),
                {"input_series": "original", "window": 100},
            ),
            (
                "ma_200",
                MovingAverage(input_col="Close"),
                {"input_series": "original", "window": 200},
            ),
            (
                "RSI",
                RelativeStrengthIndex(input_col="Close"),
                {"input_series": "original"},
            ),
            (
                "is_friday_end",
                DateFeatures(input_col="index"),
                {"input_series": "original", "func": is_friday},
            ),
            (
                "is_month_end",
                DateFeatures(input_col="index"),
                {"input_series": "original", "func": is_month_end},
            ),
            (
                "is_quarter_end",
                DateFeatures(input_col="index"),
                {"input_series": "original", "func": is_quarter_end},
            ),
            (
                "is_year_end",
                DateFeatures(input_col="index"),
                {"input_series": "original", "func": is_year_end},
            ),
            (
                "dataset_concatenator",
                DatasetConcatenator(),
                {
                    "series": [
                        "ma_10",
                        "ma_100",
                        "ma_200",
                        "RSI",
                        "is_friday_end",
                        "is_month_end",
                        "is_quarter_end",
                        "is_year_end",
                    ]
                },
            ),
            (
                "labels",
                FixedTimeHorizonLabels(input_col="Close"),
                {
                    "input_series": "original",
                    "horizon": 1,
                    "features": "dataset_concatenator",
                },
            ),
            (
                "estimator",
                RandomForestRegressor(),
                {"X": "dataset_concatenator", "y": "labels"},
            ),
        ]
    )
    pipe.fit(X=aapl_train)
    pipe.predict(X=aapl_test)
