# -*- coding: utf-8 -*-
import pandas_datareader as pdr
from sktime.pipeline.transformers import MovingAverage
from sktime.pipeline.transformers import RelativeStrengthIndex
from sktime.pipeline.transformers import DateFeatures
from sktime.pipeline.transformers import DatasetConcatenator
from sktime.pipeline.transformers import (
    is_friday,
    is_month_end,
    is_quarter_end,
    is_year_end,
)
from sktime.pipeline.pipeline import OnlineUnsupervisedPipeline

if __name__ == "__main__":
    aapl = pdr.DataReader(
        "AAPL", data_source="yahoo", start="2000-01-01", end="2021-03-31"
    )

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
                {"series": ["ma_10", "ma_100"]},
            ),
        ]
    )
    pipe.fit(X=aapl)
