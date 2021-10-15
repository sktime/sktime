#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract calendar features from datetimeindex."""
__author__ = ["Daniel Bartling"]
__all__ = ["DateTimeFeatures"]

import warnings

import numpy as np
import pandas as pd

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series

base_seasons = [
    ["child", "parent", "period", "dummy_func", "feature_scope"],
    ["year", "year", None, "year", "minimal"],
    ["quarter", "year", 365.25 / 4, "quarter", "efficient"],
    ["month", "year", 12, "month", "minimal"],
    ["week", "year", 365.25 / 7, "week_of_year", "efficient"],
    ["day", "year", 365.25, "day", "efficient"],
    ["month", "quarter", 12 / 4, "month_of_quarter", "comprehensive"],
    ["week", "quarter", 365.25 / (4 * 7), "week_of_quarter", "comprehensive"],
    ["day", "quarter", 365.25 / 4, "day_of_quarter", "comprehensive"],
    ["week", "month", 365.25 / (12 * 7), "week_of_month", "comprehensive"],
    ["day", "month", 30, "day", "efficient"],
    ["day", "week", 7, "weekday", "minimal"],
    ["hour", "day", 24, "hour", "minimal"],
    ["minute", "hour", 60, "minute", "minimal"],
    ["second", "minute", 60, "second", "minimal"],
    ["millisecond", "second", 1000, "millisecond", "minimal"],
]

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

base_seasons = pd.DataFrame(base_seasons[1:], columns=base_seasons[0])

base_seasons["fourier"] = base_seasons["child"] + "_in_" + base_seasons["parent"]
base_seasons["dummy"] = base_seasons["child"] + "_of_" + base_seasons["parent"]

base_seasons["child"] = (
    base_seasons["child"].astype("category").cat.reorder_categories(date_order)
)

flist = ["minimal", "efficient", "comprehensive"]

base_seasons["feature_scope"] = (
    base_seasons["feature_scope"].astype("category").cat.reorder_categories(flist)
)

base_seasons["feature_scope"] = pd.Categorical(
    base_seasons["feature_scope"], ordered=True
)

base_seasons["rank"] = base_seasons["child"].cat.codes

col = base_seasons["child"]
base_seasons.insert(0, "frequency", col)

base_seasons = base_seasons.replace(
    {
        "frequency": {
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


class DateTimeFeatures(_SeriesToSeriesTransformer):
    """DateTime Feature  Extraction for use in e.g. tree based models.

    Parameters
    ----------
    base_frequency : str, optional (default=None)
        Restricts selection of items to those with frequeny higher than
        the base frequency.
        E.g. if monthly data is provided, it does not make sense to derive
        weekly dummies.
        Currently has to be provided by the user due to the abundance of different
        frequencies supported by Pandas (e.g. every 4 days frequency).
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
        feature_scope: "minimal", "efficient" or "comprehensive"  - default "minimal"
        Specify how many calendar features you want to be returned.
        E.g., rarely used features like week of quarter will only be returned
        with feature_scope =  "comprehensive" .
    manual_selection:
        Manual selection of dummys. Notation is child of parent for precise notation.
        Will ignore specified feature_scope, but will still check with warning against
        a specified base_frequency
        Examples:
        * day_of_year
        * day_of_month
        * day_of_quarter
        Features year_of_year as dummy (there is no supported hierarchy above year)

    Example
    -------
    >>> from sktime.transformations.series.date import DateTimeFeatures
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = DateTimeFeatures(base_frequency="M")
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": True,
        "enforce_index_type": [pd.DatetimeIndex, pd.PeriodIndex],
        "skip-inverse-transform": True,
        "univariate-only": False,
    }

    def __init__(
        self, base_frequency=None, feature_scope="minimal", manual_selection=None
    ):

        self.base_frequency = base_frequency
        self.feature_scope = feature_scope
        self.manual_selection = manual_selection
        super(DateTimeFeatures, self).__init__()

    def transform(self, Z, X=None):
        """Transform data.

        Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame

        Returns
        -------
        Z : pd.Series, pd.DataFrame
            Transformed time series(es).
        """
        self.check_is_fitted()

        if (self.manual_selection is not None) and (
            not all(
                elem in base_seasons["dummy"].unique() for elem in self.manual_selection
            )
        ):
            raise ValueError(
                "Invalid manual_selection specified, must be in: "
                + ", ".join(base_seasons["dummy"].unique())
            )

        if self.feature_scope not in ["minimal", "efficient", "comprehensive"]:
            raise ValueError(
                "Invalid feature_scope specified,"
                + "must be in minimal,efficient,comprehensive"
                + "(minimal lowest number of variables)"
            )

        if (self.base_frequency is not None) & (
            self.base_frequency not in base_seasons["frequency"].unique()
        ):
            raise ValueError(
                "Invalid base_frequency specified, must be in: "
                + ", ".join(base_seasons["frequency"].unique())
            )

        Z = check_series(Z)
        Z = Z.copy()

        x_df = pd.DataFrame(index=Z.index)
        if isinstance(x_df.index, pd.PeriodIndex):
            x_df["date_sequence"] = Z.index.to_timestamp().astype("datetime64[ns]")
        elif isinstance(x_df.index, pd.DatetimeIndex):
            x_df["date_sequence"] = Z.index
        elif not isinstance(x_df.index, pd.DatetimeIndex):
            raise ValueError("Index type not supported")

        if self.manual_selection is None:
            if self.base_frequency is not None:
                supported = _get_supported_calendar(self.base_frequency)
                supported = supported[supported["feature_scope"] <= self.feature_scope]
                calendar_dummies = supported["dummy_func"].to_list()
            else:
                supported = base_seasons[
                    base_seasons["feature_scope"] <= self.feature_scope
                ]
                calendar_dummies = supported["dummy_func"].to_list()
        else:
            if self.base_frequency is not None:
                supported = _get_supported_calendar(self.base_frequency)
                if not all(
                    elem in supported["dummy"] for elem in self.manual_selection
                ):
                    warnings.warn(
                        "Level of selected dummy variable "
                        + " lower level than base frequency."
                    )
                calendar_dummies = supported.loc[
                    supported["dummy"].isin(self.manual_selection), "dummy_func"
                ]
            else:
                calendar_dummies = base_seasons.loc[
                    base_seasons["dummy"].isin(self.manual_selection), "dummy_func"
                ]

        df = [_calendar_dummies(x_df, dummy) for dummy in calendar_dummies]
        df = pd.concat(df, axis=1)

        Z = pd.concat([Z, df], axis=1)

        return Z


def _calendar_dummies(x, funcs):
    date_sequence = x["date_sequence"].dt
    if funcs == "week_of_year":
        # The first week of an ISO year is the first (Gregorian)
        # calendar week of a year containing a Thursday.
        # So it is possible that a week in the new year is still
        # indexed starting in last year (week 52 or 53)
        x[funcs] = date_sequence.isocalendar()["week"]
        return x[funcs]
    elif funcs == "week_of_month":
        x[funcs] = (date_sequence.day - 1) // 7 + 1
        return x[funcs]
    elif funcs == "month_of_quarter":
        x[funcs] = (np.floor(date_sequence.month / 4) + 1).astype(np.int64)
        return x[funcs]
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
        df["week_of_quarter"] = df["week"] - df["qweek"] + 1
        return df["week_of_quarter"]
    elif funcs == "millisecond":
        x[funcs] = date_sequence.microsecond * 1000
        return x[funcs]
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
        x[funcs] = values
        return x[funcs]
    else:
        x[funcs] = getattr(date_sequence, funcs)
        return x[funcs]


def _get_supported_calendar(base_frequency, base_seasons=base_seasons):
    rank = base_seasons.loc[base_seasons["frequency"] == base_frequency, "rank"].max()
    matches = base_seasons.loc[base_seasons["rank"] <= rank]
    if matches.shape[0] == 0:
        raise ValueError("Seasonality or Frequency not supported")
    return matches
