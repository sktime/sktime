#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract calendar features from datetimeindex."""
__author__ = ["danbartl", "KishManani", "VyomkeshVyas"]
__all__ = ["DateTimeFeatures"]

import warnings

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

_RAW_DUMMIES = [
    ["child", "parent", "dummy_func", "feature_scope"],
    ["year", "year", "year", "minimal"],
    ["quarter", "year", "quarter", "efficient"],
    ["month", "year", "month", "minimal"],
    ["week", "year", "week_of_year", "efficient"],
    ["day", "year", "day_of_year", "efficient"],
    ["month", "quarter", "month_of_quarter", "comprehensive"],
    ["week", "quarter", "week_of_quarter", "comprehensive"],
    ["day", "quarter", "day_of_quarter", "comprehensive"],
    ["week", "month", "week_of_month", "comprehensive"],
    ["day", "month", "day", "efficient"],
    ["day", "week", "weekday", "minimal"],
    ["hour", "day", "hour", "minimal"],
    ["hour", "week", "hour_of_week", "comprehensive"],
    ["minute", "hour", "minute", "minimal"],
    ["second", "minute", "second", "minimal"],
    ["millisecond", "second", "millisecond", "minimal"],
    ["day", "week", "is_weekend", "comprehensive"],
]


class DateTimeFeatures(BaseTransformer):
    """DateTime feature extraction, e.g., for use as exogenous data in forecasting.

    DateTimeFeatures uses a date index column and generates date features
    identifying e.g. year, week of the year, day of the week.

    Parameters
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
        Specify how many calendar features you want to be returned.
        E.g., rarely used features like week of quarter will only be returned
        with feature_scope =  "comprehensive".
        * "minimal"
        * "efficient"
        * "comprehensive"
    manual_selection: str, optional (default=None)
        Manual selection of dummys. Notation is child of parent for precise notation.
        Will ignore specified feature_scope, but will still check with warning against
        a specified ts_freq. All columns returned are integer based. Dates are presented
        in DD-MM-YYYY format below.
        Supported values:
        * None
        * quarter_of_year
            1-based index
            1-(Jan to Mar), 2-(Apr to Jun), 3-(Jul to Sep), 4-(Oct to Dec)
        * month_of_year
            1-based offset to January
            1-January,2-February,...,12-December
        * week_of_year
            1-based offset to the first week of an ISO year
        * day_of_year
            1-based offset to first of January
            1 is 01-01-YYYY, 2 is 02-01-YYYY and so on.
        * month_of_quarter
            1-based index to first month of each quarter(Jan,Apr,Jul,Oct)
            For the first quarter: 1-January, 2-February, 3-March
        * week_of_quarter
            1-based offset to first week of the quarter.
            The first/last week of the quarter may or may not include 7 days. All other
            weeks have 7 days.
            A week is taken to start on Monday.
            If the month begins on a Monday, then the first seven days upto the next
            Monday is week 1.
            Otherwise, week 1 is from the 1st of that month upto the first Monday.
            Example:
                If 01-01-YYYY is a Monday,
                Week 1 : Mon,Tue,Wed,Thu,Fri,Sat,Sun(07-01-YYYY)
                Week 2 : Mon(08-01-YYYY),Tue,...,Sun
                If 01-01-YYYY is a Thursday,
                Week 1 : Thu,Fri,Sat,Sun(04-01-YYYY)
                Week 2 : Mon(05-01-YYYY),Tue,...,Sun
        * day_of_quarter
            1-based index
        * week_of_month
            1-based index
            1 indicates the first week of the month.
            First week includes the first 7 days of the month(01-MM-YYYY to 07-MM-YYYY)
            2 indicates the second week of the month.
            Second week includes the next 7 days(08-MM-YYYY to 14-MM-YYYY) and so on.
        * day_of_month
            1-based offset to first day of each month
            1 is 01-MM-YYYY, 2 is 02-MM-YYYY and so on.
        * day_of_week
            0-based offset to Monday
            0-Monday,1-Tuesday,...,6-Sunday
        * hour_of_week
            0-based offset to Monday(00:00:00+00:00)
        * hour_of_day
            0-based offset to 00:00:00+00:00
        * minute_of_hour
            0-based offset to 00:00:00
        * second_of_minute
            0-based offset to 00:00:00
        * millisecond_of_second
            0-based offset to 00:00:00.0000
        * is_weekend
            1 indicates weekend, 0 indicates it is not a weekend
        * year (special case with no lower frequency).
    keep_original_columns :  boolean, optional, default=False
        Keep original columns in X passed to `.transform()`.

    Examples
    --------
    >>> from sktime.transformations.series.date import DateTimeFeatures
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()

    Returns columns `y`, `year`, `month_of_year`

    >>> transformer = DateTimeFeatures(ts_freq="M")
    >>> y_hat = transformer.fit_transform(y)

    Returns columns `y`, `month_of_year`

    >>> transformer = DateTimeFeatures(ts_freq="M", manual_selection=["month_of_year"])
    >>> y_hat = transformer.fit_transform(y)

    Returns columns 'y', 'year', 'quarter_of_year', 'month_of_year', 'month_of_quarter'

    >>> transformer = DateTimeFeatures(ts_freq="M", feature_scope="comprehensive")
    >>> y_hat = transformer.fit_transform(y)

    Returns columns 'y', 'year', 'quarter_of_year', 'month_of_year'

    >>> transformer = DateTimeFeatures(ts_freq="M", feature_scope="efficient")
    >>> y_hat = transformer.fit_transform(y)

    Returns columns 'y',  'year', 'month_of_year'

    >>> transformer = DateTimeFeatures(ts_freq="M", feature_scope="minimal")
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
        keep_original_columns=False,
    ):
        self.ts_freq = ts_freq
        self.feature_scope = feature_scope
        self.manual_selection = manual_selection
        self.dummies = _prep_dummies(_RAW_DUMMIES)
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
            _calendar_dummies(x_df, dummy) for dummy in calendar_dummies["dummy_func"]
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


def _calendar_dummies(x, funcs):
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


def _prep_dummies(DUMMIES):
    """Use to prepare dummy data.

    Includes defining function call names and ranking of date information based on
    frequency (e.g. year has a lower frequency than week).
    """
    DUMMIES = pd.DataFrame(DUMMIES[1:], columns=DUMMIES[0])

    date_order = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
    ]

    DUMMIES["fourier"] = DUMMIES["child"] + "_in_" + DUMMIES["parent"]
    DUMMIES["dummy"] = DUMMIES["child"] + "_of_" + DUMMIES["parent"]
    DUMMIES.loc[DUMMIES["dummy"] == "year_of_year", "dummy"] = "year"
    DUMMIES.loc[
        DUMMIES["dummy_func"] == "is_weekend", ["dummy", "fourier"]
    ] = "is_weekend"

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
