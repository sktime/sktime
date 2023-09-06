#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract peak/working hour features from datetimeindex."""
__author__ = ["ali-parizad","VyomkeshVyas"]
__all__ = ["PeakHourFeatures"]

import warnings

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

_RAW_DUMMIES = [
    ["child", "parent", "dummy_func", "feature_scope"],
    #["year", "year", "year", "minimal"],
    #["quarter", "year", "quarter", "efficient"],
    ["month", "year", "month", "comprehensive"],
    ["week", "year", "week_of_year", "comprehensive"],
    ["day", "year", "day_of_year", "comprehensive"],
    #["month", "quarter", "month_of_quarter", "comprehensive"],
    #["week", "quarter", "week_of_quarter", "comprehensive"],
    #["day", "quarter", "day_of_quarter", "comprehensive"],
    ["week", "month", "week_of_month", "efficient"],
    ["day", "month", "day", "efficient"],
    ["day", "week", "weekday", "efficient"],
    ["hour", "day", "hour", "minimal"],
    #["hour", "week", "hour_of_week", "comprehensive"],
    #["minute", "hour", "minute", "minimal"],
    #["second", "minute", "second", "minimal"],
    #["millisecond", "second", "millisecond", "minimal"],
    ##["day", "week", "is_weekend", "comprehensive"],
    ["hour", "day", "is_peak_hour", "minimal"],
    ["hour", "day", "is_working_hour", "minimal"],
] 
# TODO: Add is_working_hour for two or more intervals
# TODO: Add is_peak_week, is_peak_month, etc. if needed
# Note: For now, I commented some other features so that later we use it if needed, e.g., for adding is_peak_week

class PeakHourFeatures(BaseTransformer):
    """PeakHour feature extraction for use in e.g. tree based models.

    PeakHourFeatures uses a datetime index column and generates peak/working features
    for e.g. peak hours, working hours, peak weak, peak month, etc. 

    Parameters # TODO: docstring will be updated
    ----------
    ts_freq : str, optional (default="day")
        Restricts selection of items to those with a frequency lower than
        the frequency of the time series given by ts_freq.
        E.g. if monthly data is provided and ts_freq = ("M"), it does not make
        sense to derive dummies with higher frequency like weekly dummies.
        Has to be provided by the user due to the abundance of different
        frequencies supported by Pandas (e.g. every pandas allows freq of every 4 days).
        Interaction with other arguments:
        Used to narrow down feature selection for feature_scope, since only
        features with a frequency lower than ts_freq are considered. Will be ignored
        for the calculation of manually specified features, but when provided will
        raise a warning if manual features have a frequency higher than ts_freq.
        Only supports the following frequencies:
        * Y - year
        * Q - quarter
        * M - month
        * W - week
        * D - day
        * H - hour
        * T - minute
        * S - second
        * L - millisecond
    feature_scope: str, optional (default="minimal")
        Specify how many features you want to be returned.
        * "minimal"
        * "efficient"
        * "comprehensive"
    manual_selection: str, optional (default=None)
        Manual selection of dummys. Notation is child of parent for precise notation.
        Will ignore specified feature_scope, but will still check with warning against
        a specified ts_freq.
        Examples for possible values:
        * None
        * day_of_year
        * day_of_month
        * day_of_quarter
        * is_weekend
        * year (special case with no lower frequency).
        * is_peak_hour
        * is_working_hour
    is_peak_hour_range: list, e.g., is_peak_hour_range = {"start_peak_hour1": 6, "end_peak_hour1": 9, "start_peak_hour2": 17, "end_peak_hour2": 20}
        the peak hour range may be selcted in form of {"start_peak_hour1": 6, "end_peak_hour1": 9, "start_peak_hour2": 17, "end_peak_hour2": 20}
    is_working_hour_range: list, e.g., is_working_hour_range = {"start_working_hour": 9, "end_working_hour": 16}
        the working hour range may be selcted in form of {"start_working_hour": 9, "end_working_hour": 16}
    keep_original_columns :  boolean, optional, default=False
        Keep original columns in X passed to `.transform()`.

    Examples for hourly data (use case of 'is_peak_hour', 'is_working_hour')
    --------
    >>> from sktime.transformations.series.date import DateTimeFeatures
    >>> from sktime.datasets import load_solar
    >>> y =  load_solar(start='2022-05-01', return_full_df=True, end='2022-06-18', api_version="v4")
    >>> y = y.tz_localize(None)
    >>> y = y.asfreq("H")

    --> Example one interval (e.g., {"start_peak_hour1": 6, "end_peak_hour1": 9})for peak hours
    Returns columns `y`, 'is_peak_hour', 'is_working_hour'

    >>> transformer_peak = PeakHourFeatures(ts_freq="H", manual_selection=["is_peak_hour", 'is_working_hour'], is_peak_hour_range = {"start_peak_hour1": 6, "end_peak_hour1": 9}, is_working_hour_range = {"start_working_hour": 9, "end_working_hour": 16})
    >>> y_hat = transformer.fit_transform(y)

     --> Example two intervals (e.g., Three) for peak hours
    Returns columns `y`, 'is_peak_hour', 'is_working_hour'

    >>> transformer_peak = PeakHourFeatures(ts_freq="H", manual_selection=["is_peak_hour", 'is_working_hour'], is_peak_hour_range = {"start_peak_hour1": 6, "end_peak_hour1": 9, "start_peak_hour2": 17, "end_peak_hour2": 20}, is_working_hour_range = {"start_working_hour": 9, "end_working_hour": 16})
    >>> y_hat = transformer.fit_transform(y)   


    --> Example more than two interval (e.g.,three : {"start_peak_hour1": 1, "end_peak_hour1": 2, "start_peak_hour2": 3, "end_peak_hour2": 4, "start_peak_hour3": 7, "end_peak_hour3": 9}) for peak hours
    Returns columns `y`, 'is_peak_hour', 'is_working_hour'

    >>> transformer_peak = PeakHourFeatures(ts_freq="H", manual_selection=["is_peak_hour",'is_working_hour'], is_peak_hour_range = {"start_peak_hour1": 1, "end_peak_hour1": 2, "start_peak_hour2": 3, "end_peak_hour2": 4, "start_peak_hour3": 7, "end_peak_hour3": 9}, is_working_hour_range = {"start_working_hour": 9, "end_working_hour": 16})
    >>> y_hat = transformer.fit_transform(y)

    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "enforce_index_type": [pd.DatetimeIndex, pd.PeriodIndex],
        "skip-inverse-transform": True,
        "python_dependencies": "pandas>=1.2.0",  # from DateTimeProperties
    }

    def __init__(
        self,
        ts_freq=None,
        feature_scope="minimal",
        manual_selection=None,
        is_peak_hour_range=None,
        is_working_hour_range=None,
        keep_original_columns=False,
    ):
        self.ts_freq = ts_freq
        self.feature_scope = feature_scope
        self.manual_selection = manual_selection
        self.is_peak_hour_range = is_peak_hour_range
        self.is_working_hour_range = is_working_hour_range
        self.dummies = _prep_dummies(self, _RAW_DUMMIES)
        self.keep_original_columns = keep_original_columns

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        _check_ts_freq(self.ts_freq, self.dummies)
        _check_feature_scope(self.feature_scope)
        _check_manual_selection(self.manual_selection, self.dummies)
        _check_is_peak_hour_range(self.is_peak_hour_range)
        _check_is_working_hour_range(self.is_working_hour_range)

        if isinstance(X.index, pd.MultiIndex):
            time_index = X.index.get_level_values(-1)
        else:
            time_index = X.index

        x_df = pd.DataFrame(index=X.index)
        if isinstance(time_index, pd.PeriodIndex):
            x_df["date_sequence"] = time_index.to_timestamp()
        elif isinstance(time_index, pd.DatetimeIndex):
            x_df["date_sequence"] = time_index
        else:
            raise ValueError("Index type not supported")

        if self.manual_selection is None:
            if self.ts_freq is not None:
                supported = _get_supported_calendar(self.ts_freq, DUMMIES=self.dummies)
                supported = supported[supported["feature_scope"] <= self.feature_scope]
                calendar_dummies = supported[["dummy_func", "dummy"]]
            else:
                supported = self.dummies[
                    self.dummies["feature_scope"] <= self.feature_scope
                ]
                calendar_dummies = supported[["dummy_func", "dummy"]]
        else:
            if self.ts_freq is not None:
                supported = _get_supported_calendar(self.ts_freq, DUMMIES=self.dummies)
                if not all(
                    elem in supported["dummy"] for elem in self.manual_selection
                ):
                    warnings.warn(
                        "Level of selected dummy variable "
                        + " lower level than base ts_frequency.",
                        stacklevel=2,
                    )
                calendar_dummies = self.dummies.loc[
                    self.dummies["dummy"].isin(self.manual_selection),
                    ["dummy_func", "dummy"],
                ]
            else:
                calendar_dummies = self.dummies.loc[
                    self.dummies["dummy"].isin(self.manual_selection),
                    ["dummy_func", "dummy"],
                ]

        df = [
            _calendar_dummies(self, x_df, dummy) for dummy in calendar_dummies["dummy_func"]
        ]


        df = pd.concat(df, axis=1)
        df.columns = calendar_dummies["dummy"]

        if self.keep_original_columns:
            Xt = pd.concat([X, df], axis=1, copy=True)
        else:
            # Remove the name `"dummy"` from column index.
            Xt = df.rename_axis(None, axis="columns")

        return Xt

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"feature_scope": "minimal"}
        params2 = {"feature_scope": "efficient", "keep_original_columns": True}
        params3 = {"manual_selection": ["day_of_year", "day_of_month"]}
        return [params1, params2, params3]


def _check_manual_selection(manual_selection, DUMMIES):
    if (manual_selection is not None) and (
        not all(elem in DUMMIES["dummy"].unique() for elem in manual_selection)
    ):
        raise ValueError(
            "Invalid manual_selection specified, must be in: "
            + ", ".join(DUMMIES["dummy"].unique())
        )


def _check_is_peak_hour_range(is_peak_hour_range):
    if (is_peak_hour_range is not None): 
        for idx in range (int(len(is_peak_hour_range)/2)):
            start_peak_hour = is_peak_hour_range[f'start_peak_hour{idx+1}']
            end_peak_hour = is_peak_hour_range[f'end_peak_hour{idx+1}']

            if start_peak_hour < 0 or end_peak_hour > 23:
                print(
                    f"you selected start_peak_hour{idx+1} = {start_peak_hour}, end_peak_hour{idx+1} = {end_peak_hour}"
                )
                raise ValueError(
                    "Invalid is_peak_hour_range specified,"
                    + "must be in range of 0 - 23 hour"
                    + "(min peak hour = 0, max peak hour = 23)"
                )
            if start_peak_hour > end_peak_hour: 
                print(
                    f"you selected start_peak_hour{idx+1} = {start_peak_hour}, end_peak_hour{idx+1} = {end_peak_hour}"
                )
                raise ValueError(
                    "Invalid is_peak_hour_range specified,"
                    + "min is_peak_hour_range (start_peak_hour) must be less than max is_peak_hour_range (end_peak_hour)"
                    + "(min peak hour = 0, max peak hour = 23)"
                )


def _check_is_working_hour_range(is_working_hour_range): 
    if (is_working_hour_range is not None): 
        if is_working_hour_range["start_working_hour"] < 0 or is_working_hour_range["end_working_hour"] > 23:
            print(
                f"you selected min(is_working_hour_range) = {is_working_hour_range['start_working_hour']}, max(is_working_hour_range) = {is_working_hour_range['end_working_hour']}"
            )
            raise ValueError(
                "Invalid is_working_hour_range specified,"
                + "must be in range of 0 - 23 hour"
                + "(min peak hour = 0, max peak hour = 23)"
            )
        if is_working_hour_range["start_working_hour"] > is_working_hour_range["end_working_hour"]:
            print(
                f"you selected min(is_working_hour_range) = {is_working_hour_range['start_working_hour']}, max(is_working_hour_range) = {is_working_hour_range['end_working_hour']}"
            )
            raise ValueError(
                "Invalid is_working_hour_range specified,"
                + "min is_working_hour_range (peak_hour_start) must be less than max is_working_hour_range (peak_hour_end)"
                + "(min peak hour = 0, max peak hour = 23)"
            )
    

def _check_feature_scope(feature_scope):
    if feature_scope not in ["minimal", "efficient", "comprehensive"]:
        raise ValueError(
            "Invalid feature_scope specified,"
            + "must be in minimal,efficient,comprehensive"
            + "(minimal lowest number of variables)"
        )


def _check_ts_freq(ts_freq, DUMMIES):
    if (ts_freq is not None) & (ts_freq not in DUMMIES["ts_frequency"].unique()):
        raise ValueError(
            "Invalid ts_freq specified, must be in: "
            + ", ".join(DUMMIES["ts_frequency"].unique())
        )


def _calendar_dummies(self, x, funcs):
    date_sequence = x["date_sequence"].dt
    if funcs == "week_of_year":
        # The first week of an ISO year is the first (Gregorian)
        # calendar week of a year containing a Thursday.
        # So it is possible that a week in the new year is still
        # indexed starting in last year (week 52 or 53)
        cd = date_sequence.isocalendar()["week"]
    elif funcs == "week_of_month":
        cd = (date_sequence.day - 1) // 7 + 1
    elif funcs == "month_of_quarter":
        cd = (date_sequence.month.astype(np.int64) + 2) % 3 + 1
    elif funcs == "week_of_quarter":
        col_names = x.columns
        x_columns = col_names.intersection(["year", "quarter", "week"]).to_list()
        x_columns.append("date_sequence")
        df = x.copy(deep=True)
        df = df[x_columns]
        if "year" not in x_columns:
            df["year"] = df["date_sequence"].dt.year
        if "quarter" not in x_columns:
            df["quarter"] = df["date_sequence"].dt.quarter
        if "week" not in x_columns:
            df["week"] = df["date_sequence"].dt.isocalendar()["week"]
        df["qdate"] = (
            df["date_sequence"] + pd.tseries.offsets.DateOffset(days=1)
        ) - pd.tseries.offsets.QuarterBegin(startingMonth=1)
        df["qweek"] = df["qdate"].dt.isocalendar()["week"]
        df.loc[(df["quarter"] == 1) & (df["week"] < 52), "qweek"] = 0
        cd = df["week"] - df["qweek"] + 1
    elif funcs == "millisecond":
        cd = date_sequence.microsecond * 1000
    elif funcs == "day_of_quarter":
        quarter = date_sequence.quarter
        quarter_start = pd.DatetimeIndex(
            date_sequence.year.map(str)
            + "-"
            + (3 * quarter - 2).map(int).map(str)
            + "-01"
        )
        values = (
            (x["date_sequence"] - quarter_start) / pd.to_timedelta("1D") + 1
        ).astype(int)
        cd = values
    elif funcs == "hour_of_week":
        cd = date_sequence.day_of_week * 24 + date_sequence.hour
    elif funcs == "is_weekend":
        cd = date_sequence.day_of_week > 4
    elif (funcs == "is_peak_hour") & (self.is_peak_hour_range is not None):
        cd_combined= []
        for idx in range (int(len(self.is_peak_hour_range)/2)):
            cd_temp = (date_sequence.hour >= self.is_peak_hour_range[f'start_peak_hour{idx+1}']) & (date_sequence.hour <= self.is_peak_hour_range[f'end_peak_hour{idx+1}'])  
            cd_combined.append(cd_temp)
        cd_combined_concat = pd.concat(cd_combined, axis = 1)
        cd = cd_combined_concat.any(axis=1)

    elif (funcs == "is_working_hour")  & (self.is_working_hour_range is not None):
        cd = (date_sequence.hour >= self.is_working_hour_range["start_working_hour"]) & (date_sequence.hour <= self.is_working_hour_range["end_working_hour"])

    else:
        cd = getattr(date_sequence, funcs)
    cd = pd.DataFrame(cd)
    cd = cd.rename(columns={cd.columns[0]: funcs})
    cd[funcs] = np.int64(cd[funcs])
    return cd


def _get_supported_calendar(ts_freq, DUMMIES):
    rank = DUMMIES.loc[DUMMIES["ts_frequency"] == ts_freq, "rank"].max()
    matches = DUMMIES.loc[DUMMIES["rank"] <= rank]
    if matches.shape[0] == 0:
        raise ValueError("Seasonality or Frequency not supported")
    return matches

def _prep_dummies(self, DUMMIES):
    """Use to prepare dummy data.

    Includes defining function call names and ranking of date information based on
    frequency (e.g. year has a lower frequency than week).
    """
    DUMMIES = pd.DataFrame(DUMMIES[1:], columns=DUMMIES[0])
# Note: For now, I commented some other features so that later we use it if needed, e.g., for adding is_peak_week
    date_order = [
        #"year",
        #"quarter",
        "month",
        "week",
        "day",
        "hour",
        #"minute",
        #"second",
        #"millisecond",
    ]

    DUMMIES["fourier"] = DUMMIES["child"] + "_in_" + DUMMIES["parent"]
    DUMMIES["dummy"] = DUMMIES["child"] + "_of_" + DUMMIES["parent"]
    #DUMMIES.loc[DUMMIES["dummy"] == "year_of_year", "dummy"] = "year"
    #DUMMIES.loc[
    #    DUMMIES["dummy_func"] == "is_weekend", ["dummy", "fourier"]
   #] = "is_weekend"

    if self.is_peak_hour_range is not None:
        DUMMIES.loc[
            DUMMIES["dummy_func"] == "is_peak_hour", ["dummy", "fourier"]
        ] = "is_peak_hour"
    elif self.is_peak_hour_range is None:
        DUMMIES = DUMMIES[DUMMIES.dummy_func != "is_peak_hour"]


    if self.is_working_hour_range is not None:
        DUMMIES.loc[
            DUMMIES["dummy_func"] == "is_working_hour", ["dummy", "fourier"]
        ] = "is_working_hour"
    elif self.is_working_hour_range is None:
        DUMMIES = DUMMIES[DUMMIES.dummy_func != "is_working_hour"]

    DUMMIES["child"] = (
        DUMMIES["child"].astype("category").cat.reorder_categories(date_order)
    )

    flist = ["minimal", "efficient", "comprehensive"]

    DUMMIES["feature_scope"] = (
        DUMMIES["feature_scope"].astype("category").cat.reorder_categories(flist)
    )

    DUMMIES["feature_scope"] = pd.Categorical(DUMMIES["feature_scope"], ordered=True)

    DUMMIES["rank"] = DUMMIES["child"].cat.codes

    col = DUMMIES["child"]
    DUMMIES.insert(0, "ts_frequency", col)

    DUMMIES = DUMMIES.replace(
        {
            "ts_frequency": {
                "year": "Y",
                "quarter": "Q",
                "month": "M",
                "week": "W",
                "day": "D",
                "hour": "H",
                "minute": "T",
                "second": "S",
                "millisecond": "L",
            }
        }
    )

    return DUMMIES
