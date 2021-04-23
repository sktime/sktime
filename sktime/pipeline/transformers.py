# -*- coding: utf-8 -*-
import mlfinlab as ml
import pandas as pd
import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.transformations.base import BaseTransformer

from sktime.transformations.base import Series
import logging


class _ComplexToSeriesTransformer(BaseTransformer):
    """Transformer base class for series to series transforms"""

    def transform(self, Z: Series, X=None) -> Series:
        raise NotImplementedError("abstract method")


class _ComplexToTabular(BaseTransformer):
    """Transformer base class for series to series transforms"""

    def transform(self, Z: Series, X=None) -> Series:
        raise NotImplementedError("abstract method")


class DollarBars(_SeriesToSeriesTransformer):
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

        return dollar_bars

    def fit_transform(self, X):
        """
        Dummy method, calls transform
        """
        return self.transform(X)

    def fit(self):
        logging.warning("testing")
        # raise NotImplementedError

    def update(self):
        raise NotImplementedError


class CUSUM(_SeriesToSeriesTransformer):
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

        return self.transform(input_series)

    def transform(self, input_series):
        cusum_events = ml.filters.cusum_filter(
            input_series[self._price_col], threshold=self._threshold
        )
        return cusum_events

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class DailyVol(_SeriesToSeriesTransformer):
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
        return daily_vol

    def fit_transform(self, input_series):
        """
        Dummy method, calls transform
        """
        return self.transform(input_series)

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class TrippleBarrierEvents(_ComplexToSeriesTransformer):
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

        return self.transform(input_series, change_points, target)

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

        return triple_barrier_events

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class TrippleBarrierLabels(_ComplexToSeriesTransformer):
    _tags = {"fit-in-transform": True}

    def __repr__(self):
        return repr(
            "Calculates labels from triple barrier events. \
            Inputs: Triple Barrier Events (pd dataframe) \
                Output: pd Dataframe"
        )

    def __init__(self, price_col):
        self._price_col = price_col

    def fit_transform(self, input_series, events):
        return self.transform(input_series, events)

    def transform(self, input_series, events):

        triple_labels = ml.labeling.get_bins(events, input_series[self._price_col])

        return triple_labels

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class BuildDataset(_ComplexToTabular):
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
        return self.transform(input_dataset, labels)

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

        return dataset

    def fit(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


# -----------------------------------------------------
class MovingAverage(_SeriesToSeriesTransformer):
    def __init__(self, input_col):
        self._input_col = input_col

    def transform(self, input_series, window):
        """
        Parameters
        ----------
        input_series : pandas Dataframe
        window : int
        """
        result = input_series[self._input_col].rolling(window).mean()
        result.name = f"MA_{window}"
        result = input_series[self._input_col].rolling(window).mean()
        return result

    def fit_transform(self, input_series, window):
        return self.transform(input_series, window)


class RelativeStrengthIndex(_SeriesToSeriesTransformer):
    def __init__(self, input_col):
        self._input_col = input_col

    def transform(self, input_series):
        delta = input_series[self._input_col].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        rs.name = "RSI"
        return rs

    def fit_transform(self, input_series):
        return self.transform(input_series)


class DateFeatures(_SeriesToSeriesTransformer):
    def __init__(self, input_col):
        self._input_col = input_col

    def transform(self, input_series, func):
        result = None
        if self._input_col == "index":
            result = pd.Series(data=input_series.index, index=input_series.index)
        else:
            result = pd.Series(
                data=input_series[self._input_col], index=input_series.index
            )
        result.name = f"df_{func.__name__}"
        result = result.apply(func)

        return result

    def fit_transform(self, input_series, func):
        return self.transform(input_series, func)


class FixedTimeHorizonLabels(_SeriesToSeriesTransformer):
    def __init__(self, input_col):
        self._input_col = input_col

    def transform(self, input_series, horizon, features):
        labels = input_series[self._input_col].shift(horizon)
        labels.name = "labels"
        dts = pd.concat([features, labels], axis=1).dropna()
        return dts["labels"]

    def fit_transform(self, input_series, horizon, features):
        return self.transform(input_series, horizon, features)


class DatasetConcatenator(_SeriesToSeriesTransformer):
    """Concatenates pandas series by index and removes NA values"""

    def transform(self, series):

        dts = pd.concat(series, axis=1)
        dts.dropna(inplace=True)
        return dts

    def fit_transform(self, series):
        return self.transform(series)


def is_friday(dt):
    if dt.weekday() == 4:
        return True
    else:
        return False


def is_month_end(dt):
    return dt.is_month_end


def is_quarter_end(dt):
    return dt.is_quarter_end


def is_year_end(dt):
    return dt.is_year_end
