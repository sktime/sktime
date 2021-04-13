# -*- coding: utf-8 -*-
import mlfinlab as ml
import pandas as pd
import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.transformations.base import BaseTransformer

from sktime.transformations.base import Series
import logging
from sktime.pipeline.base import _NonSequentialPipelineStepResultsMixin


class _ComplexToSeriesTransformer(BaseTransformer):
    """Transformer base class for series to series transforms"""

    def transform(self, Z: Series, X=None) -> Series:
        raise NotImplementedError("abstract method")


class _ComplexToTabular(BaseTransformer):
    """Transformer base class for series to series transforms"""

    def transform(self, Z: Series, X=None) -> Series:
        raise NotImplementedError("abstract method")


class DollarBars(_SeriesToSeriesTransformer, _NonSequentialPipelineStepResultsMixin):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "Calculates Dollar Bars. Inputs: closing price and traded volume as pd dataframe \
             Output pd dataframe"
        )

    def transform(self, X):
        """
        Parameters:
        -----------
        X : pandas dataframe
            Index (datetime), price, volume
        """
        dollar_bars = ml.data_structures.get_dollar_bars(
            file_path_or_df=X, threshold=1e10
        )
        dollar_bars = dollar_bars.set_index("date_time")
        dollar_bars.index = pd.to_datetime(dollar_bars.index)
        dollar_bars = dollar_bars.sort_index(ascending=True)
        idx = dollar_bars.index.duplicated(keep="first")
        dollar_bars = dollar_bars.loc[~idx]

        self.step_result = dollar_bars

        return self

    def fit_transform(self, X):
        """
        Dummy method, calls transform
        """
        self.transform(X)

    def fit(self):
        logging.warning("testing")
        # raise NotImplementedError

    def update(self):
        raise NotImplementedError


class CUSUM(_SeriesToSeriesTransformer, _NonSequentialPipelineStepResultsMixin):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "CUSUM Fulter. Inputs closing price (pd Series). \
            Outputs: Timestamps of events (Datetime Index)"
        )

    def __init__(self, price_col, threshold=-0.1):
        """
        Paramters
        ---------
        price_col : string
            name of column in pandas dataframe
        """
        self._threshold = threshold
        self._price_col = price_col

    def fit_transform(self, input_series):
        """
        Dummy method, calls transform
        """
        self.transform(input_series)

    def transform(self, input_series):
        cusum_events = ml.filters.cusum_filter(
            input_series[self._price_col], threshold=self._threshold
        )
        self.step_result = cusum_events
        return self

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class DailyVol(_SeriesToSeriesTransformer, _NonSequentialPipelineStepResultsMixin):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "Calculates daily volatility. Inputs: closing prices (pd Series). \
            Output: daily volatility (pd Series)"
        )

    def __init__(self, price_col, lookback):
        self._lookback = lookback
        self._price_col = price_col

    def transform(self, input_series):
        daily_vol = ml.util.volatility.get_daily_vol(
            close=input_series[self._price_col], lookback=self._lookback
        )
        self.step_result = daily_vol
        return self

    def fit_transform(self, input_series):
        """
        Dummy method, calls transform
        """
        self.transform(input_series)

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class TrippleBarrierEvents(
    _ComplexToSeriesTransformer, _NonSequentialPipelineStepResultsMixin
):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "Calculates Triple Barrier Events. \
        Inputs: 1. closing prices (dollar bars output, pd series) \
            2. change points (CUSUM or other, datetime index) \
                Output: pd DataFrame with annotations"
        )

    def __init__(
        self,
        price_col,
        num_days,
        pt_sl=None,
        target_barrier_multiple=2,
        min_return=0.005,
        num_threads=12,
    ):
        self._price_col = price_col
        self._num_days = num_days
        if pt_sl is None:
            self._pt_sl = [1, 1]
        else:
            self._pt_sl = pt_sl
        self._target_barrier_multiple = target_barrier_multiple
        self._min_return = min_return
        self._num_threads = num_threads

    def fit_transform(self, input_series, change_points, target):
        """
        Dummy method, calls transform
        """
        self.transform(input_series, change_points, target)

    def transform(self, input_series, change_points, target):
        vertical_barriers = ml.labeling.add_vertical_barrier(
            t_events=change_points,
            close=input_series[self._price_col],
            num_days=self._num_days,
        )
        triple_barrier_events = ml.labeling.get_events(
            close=input_series[self._price_col],
            t_events=change_points,
            pt_sl=self._pt_sl,  # sets the width of the barriers
            target=target * self._target_barrier_multiple,  # twice the daily vol
            min_ret=self._min_return,
            num_threads=self._num_threads,
            vertical_barrier_times=vertical_barriers,
            side_prediction=None,
            verbose=False,
        )
        self.step_result = triple_barrier_events

        return self

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class TrippleBarrierLabels(
    _ComplexToSeriesTransformer, _NonSequentialPipelineStepResultsMixin
):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "Calculates labels from triple barrier events. \
            Inputs: Triple Barrier Events (pd dataframe) \
                Output: pd Dataframe"
        )

    def __init__(self, price_col):
        self._price_col = price_col

    def fit_transform(self, triple_barrier_events, prices):
        """Dummy method, calls transform"""
        self.transform(triple_barrier_events, prices)

    def transform(self, triple_barrier_events, prices):

        triple_labels = ml.labeling.get_bins(
            triple_barrier_events, prices[self._price_col]
        )

        self.step_result = triple_labels

        return self

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class BuildDataset(_ComplexToTabular, _NonSequentialPipelineStepResultsMixin):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "Compiles dataset. Inputs: Triple Barier Events and closing prices \
            Outputs: features dataset used for training"
        )

    def __init__(self, price_col, labels_col, lookback):
        self._lookback = lookback
        self._price_col = price_col
        self._labels_col = labels_col

    def fit_transform(self, input_dataset, labels):
        """Dummy method, calls transform"""
        self.transform(input_dataset, labels)

    def transform(self, input_dataset, labels):
        col_names = [f"feature_{i}" for i in np.arange(self._lookback)]
        dataset = pd.DataFrame(columns=col_names)
        for i in np.arange(labels.shape[0]):
            values = (
                input_dataset[self._price_col]
                .iloc[
                    input_dataset.index.get_loc(labels.index[i])
                    - self._lookback : input_dataset.index.get_loc(labels.index[i])
                ]
                .values
            )
            data = pd.DataFrame(values.reshape(1, self._lookback), columns=col_names)
            dataset = dataset.append(data)
        dataset[self._labels_col] = labels[self._labels_col].values
        dataset[self._labels_col] = dataset[self._labels_col].replace(-1, 0)

        self.step_result = dataset

        return self

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
