#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract calendar features from datetimeindex."""
__author__ = ["danbartl"]
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
    ["minute", "hour", "minute", "minimal"],
    ["second", "minute", "second", "minimal"],
    ["millisecond", "second", "millisecond", "minimal"],
]


class DateTimeFeatures(BaseTransformer):
    """DateTime Feature  Extraction for use in e.g. tree based models.

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
        a specified ts_freq.
        Examples for Possible values:
        * None
        * day_of_year
        * day_of_month
        * day_of_quarter
        * year (special case with no lower frequency).

    Examples
    --------
    >>> from sktime.transformations.series.date import DateTimeFeatures
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = DateTimeFeatures(ts_freq="M")
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "enforce_index_type": [pd.DatetimeIndex, pd.PeriodIndex],
        "skip-inverse-transform": True,
    }

    def __init__(self, ts_freq=None, feature_scope="minimal", manual_selection=None):

        self.ts_freq = ts_freq
        self.feature_scope = feature_scope
        self.manual_selection = manual_selection
        self.dummies = _prep_dummies(_RAW_DUMMIES)
        super(DateTimeFeatures, self).__init__()

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

        Z = X.copy()

        x_df = pd.DataFrame(index=Z.index)
        if isinstance(x_df.index, pd.PeriodIndex):
            x_df["date_sequence"] = Z.index.to_timestamp().astype("datetime64[ns]")
        elif isinstance(x_df.index, pd.DatetimeIndex):
            x_df["date_sequence"] = Z.index
        elif not isinstance(x_df.index, pd.DatetimeIndex):
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
                        + " lower level than base ts_frequency."
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
        if self.manual_selection is not None:
            df = df[self.manual_selection]

        Xt = pd.concat([Z, df], axis=1)

        return Xt


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
        cd = (np.floor(date_sequence.month / 4) + 1).astype(np.int64)
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

    Includes defining function call names and ranking
    of date information based on frequency (e.g. year
    has a lower frequency than week).
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
