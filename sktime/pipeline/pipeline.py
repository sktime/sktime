# -*- coding: utf-8 -*-
import mlfinlab as ml
import pandas as pd
import numpy as np
from sktime.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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
    def __repr__(self):
        return repr(
            "Calculates Dollar Bars. Inputs: closing price and traded volume as pd dataframe \
             Output pd dataframe"
        )

    def fit(self, X):
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

        self._fit_result = dollar_bars

        return self

    def transform(self):
        logging.warning("testing")
        # raise NotImplementedError

    def update(self):
        raise NotImplementedError


class CUSUM(_SeriesToSeriesTransformer):
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

    def fit(self, input_series):
        cusum_events = ml.filters.cusum_filter(
            input_series[self._price_col], threshold=self._threshold
        )
        self._fit_result = cusum_events
        return self

    def transform(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class DailyVol(_SeriesToSeriesTransformer):
    def __repr__(self):
        return repr(
            "Calculates daily volatility. Inputs: closing prices (pd Series). \
            Output: daily volatility (pd Series)"
        )

    def __init__(self, price_col, lookback):
        self._lookback = lookback
        self._price_col = price_col

    def fit(self, input_series):
        daily_vol = ml.util.volatility.get_daily_vol(
            close=input_series[self._price_col], lookback=self._lookback
        )
        self._fit_result = daily_vol
        return self

    def transform(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class TrippleBarrierEvents(_ComplexToSeriesTransformer):
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

    def fit(self, prices, change_points, daily_vol):
        vertical_barriers = ml.labeling.add_vertical_barrier(
            t_events=change_points,
            close=prices[self._price_col],
            num_days=self._num_days,
        )
        triple_barrier_events = ml.labeling.get_events(
            close=prices[self._price_col],
            t_events=change_points,
            pt_sl=self._pt_sl,  # sets the width of the barriers
            target=daily_vol * self._target_barrier_multiple,  # twice the daily vol
            min_ret=self._min_return,
            num_threads=self._num_threads,
            vertical_barrier_times=vertical_barriers,
            side_prediction=None,
            verbose=False,
        )
        self._fit_result = triple_barrier_events

        return self

    def transform(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class TrippleBarrierLabels(_ComplexToSeriesTransformer):
    def __repr__(self):
        return repr(
            "Calculates labels from triple barrier events. \
            Inputs: Triple Barrier Events (pd dataframe) \
                Output: pd Dataframe"
        )

    def __init__(self, price_col):
        self._price_col = price_col

    def fit(self, triple_barrier_events, prices):

        triple_labels = ml.labeling.get_bins(
            triple_barrier_events, prices[self._price_col]
        )

        self._fit_result = triple_labels

        return self

    def transform(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class BuildDataset(_ComplexToTabular):
    def __repr__(self):
        return repr(
            "Compiles dataset. Inputs: Triple Barier Events and closing prices \
            Outputs: features dataset used for training"
        )

    def __init__(self, price_col, labels_col, lookback):
        self._lookback = lookback
        self._price_col = price_col
        self._labels_col = labels_col

    def fit(self, input_dataset, labels):
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

        self._fit_result = dataset

        return self

    def transform(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class Estimator(BaseEstimator):
    def __init__(
        self,
        estimator,
        param_grid,
        samples_col_name,
        labels_col_name,
        scoring="neg_log_loss",
        shuffle=False,
        test_size=0.25,
        n_splits=5,
        pct_embargo=0.01,
    ):
        self._samples_col_name = samples_col_name
        self._labels_col_name = labels_col_name
        self._n_splits = n_splits
        self._pct_embargo = pct_embargo
        self._shuffle = shuffle
        self._test_size = test_size
        self._estimator = estimator
        self._param_grid = param_grid
        self._scoring = scoring

    def fit(self, X, y, samples):
        train_idx, test_idx = train_test_split(
            np.arange(X.shape[0]), shuffle=self._shuffle, test_size=self._test_size
        )
        cv_gen = ml.cross_validation.PurgedKFold(
            samples_info_sets=samples[self._samples_col_name].iloc[train_idx],
            n_splits=self._n_splits,
            pct_embargo=self._pct_embargo,
        )
        gs = GridSearchCV(
            estimator=self._estimator,
            param_grid=self._param_grid,
            scoring=self._scoring,
            cv=cv_gen,
        )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx][self._labels_col_name]

        trained_estimator = gs.fit(X_train, y_train)
        self._fit_result = trained_estimator

        return self

    def predict(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class OnlineUnsupervisedPipeline(BaseEstimator):
    """
    Parameters
    ----------
    steps : array of lists
        list comprised of three elements:
            1. name of step (string),
            2. algorithm (object),
            3. input (dictionary) key value paris for fit() method of algorithm
    interface: sting
        `simple`(default) or `advanced`.
            If `simple` `fit` method of the object is called.
            If `advanced` name of method can be specified like this:
            (`name of step`, `algorithm`, `input`), where input:
        input_dict = [
            {
                function: name of function to be called (fit, transform, predict ...),
                arguments: key value pairs(value is name of step) ,
            },
            {
                function: name of function to be called,
                arguments: key value pairs(value is name of step),
            },
            ......
        ]

    """

    def __init__(self, steps, interface="simple"):
        self._steps = steps
        self._results_dict = {}
        self._interface = interface

    def _iter(self):
        for name, alg, arguments in self._steps:
            yield name, alg, arguments

    def _check_arguments(self, arguments):
        """
        Checks arguments for consistency.

        Replace key word 'original' with X

        Replace string step name with step fit return result

        Parameters
        ----------
        arguments : dictionary
            key-value for fit() method of algorithm
        """
        if arguments is None:
            return arguments
        for key, value in arguments.items():
            if value == "original":
                arguments[key] = self._X

            if value in self._results_dict:
                arguments[key] = self._results_dict[value]._fit_result

        return arguments

    def fit(self, X):
        self._X = X
        for name, alg, arguments in self._iter():
            if self._interface == "simple":
                arguments = self._check_arguments(arguments)
                self._results_dict[name] = alg.fit(**arguments)
                # print(f"{name} fitted")
            if self._interface == "advanced":
                for arg in arguments:
                    func_name = arg["function"]
                    arguments = self._check_arguments(arg["arguments"])
                    # execute function
                    if arguments is not None:
                        self._results_dict[name] = getattr(alg, func_name)(**arguments)
                    else:
                        self._results_dict[name] = getattr(alg, func_name)()

        return self


if __name__ == "__main__":

    pipe = OnlineUnsupervisedPipeline(
        steps=[
            ("dollar_bars", DollarBars(), {"X": "original"}),
            ("cusum", CUSUM(price_col="close"), {"input_series": "dollar_bars"}),
            (
                "daily_vol",
                DailyVol(price_col="close", lookback=5),
                {"input_series": "dollar_bars"},
            ),
            (
                "triple_barrier_events",
                TrippleBarrierEvents(price_col="close", num_days=5),
                {
                    "prices": "dollar_bars",
                    "change_points": "cusum",
                    "daily_vol": "daily_vol",
                },
            ),
            (
                "labels",
                TrippleBarrierLabels(price_col="close"),
                {
                    "triple_barrier_events": "triple_barrier_events",
                    "prices": "dollar_bars",
                },
            ),
            (
                "build_dataset",
                BuildDataset(price_col="close", labels_col="bin", lookback=20),
                {"input_dataset": "dollar_bars", "labels": "labels"},
            ),
            (
                "estimator",
                Estimator(
                    estimator=BaggingClassifier(
                        base_estimator=DecisionTreeClassifier(
                            max_depth=5, random_state=1
                        )
                    ),
                    param_grid={
                        "n_estimators": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25]
                    },
                    samples_col_name="t1",
                    labels_col_name="bin",
                ),
                {
                    "X": "build_dataset",
                    "y": "labels",
                    "samples": "triple_barrier_events",
                },
            ),
        ]
    )
    pipe.fit(X="sktime/pipeline/curated_tick_data.csv")

    pipeAdvanced = OnlineUnsupervisedPipeline(
        steps=[
            (
                "dollar_bars",
                DollarBars(),
                [
                    {"function": "fit", "arguments": {"X": "original"}},
                    {"function": "transform", "arguments": None},
                ],
            )
            # ("cusum", CUSUM(price_col="close"), {"input_series": "dollar_bars"}),
            # (
            #     "daily_vol",
            #     DailyVol(price_col="close", lookback=5),
            #     {"input_series": "dollar_bars"},
            # ),
            # (
            #     "triple_barrier_events",
            #     TrippleBarrierEvents(price_col="close", num_days=5),
            #     {
            #         "prices": "dollar_bars",
            #         "change_points": "cusum",
            #         "daily_vol": "daily_vol",
            #     },
            # ),
            # (
            #     "labels",
            #     TrippleBarrierLabels(price_col="close"),
            #     {
            #         "triple_barrier_events": "triple_barrier_events",
            #         "prices": "dollar_bars",
            #     },
            # ),
            # (
            #     "build_dataset",
            #     BuildDataset(price_col="close", labels_col="bin", lookback=20),
            #     {"input_dataset": "dollar_bars", "labels": "labels"},
            # ),
            # (
            #     "estimator",
            #     Estimator(
            #         estimator=BaggingClassifier(
            #             base_estimator=DecisionTreeClassifier(
            #                 max_depth=5, random_state=1
            #             )
            #         ),
            #         param_grid={
            #             "n_estimators": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25]
            #         },
            #         samples_col_name="t1",
            #         labels_col_name="bin",
            #     ),
            #     {
            #         "X": "build_dataset",
            #         "y": "labels",
            #         "samples": "triple_barrier_events",
            #     },
            # ),
        ],
        interface="advanced",
    )
    pipeAdvanced.fit(X="sktime/pipeline/curated_tick_data.csv")
