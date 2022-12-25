# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract calendar features from datetimeindex."""
from __future__ import annotations

__author__ = ["KishManani"]

import warnings

import pandas as pd

from sktime.transformations.series.date import (
    DateTimeFeatures as DateTimeFeaturesSeries,
)
from sktime.transformations.series.date import (
    _calendar_dummies,
    _check_feature_scope,
    _check_manual_selection,
    _check_ts_freq,
    _get_supported_calendar,
)


class DateTimeFeatures(DateTimeFeaturesSeries):
    """DateTime feature extraction for use in e.g. tree based models.

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
        Examples for possible values:
        * None
        * day_of_year
        * day_of_month
        * day_of_quarter
        * is_weekend
        * year (special case with no lower frequency).

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
        "scitype:transform-input": "Panel",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Panel",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "enforce_index_type": [pd.DatetimeIndex, pd.PeriodIndex],
        "skip-inverse-transform": True,
    }

    def __init__(
        self,
        ts_freq: str | None = None,
        feature_scope: str | None = "minimal",
        manual_selection: str | None = None,
    ):
        super(DateTimeFeatures, self).__init__(
            ts_freq=ts_freq,
            feature_scope=feature_scope,
            manual_selection=manual_selection,
        )

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
        if isinstance(x_df.index.levels[-1], pd.PeriodIndex):
            x_df["date_sequence"] = (
                Z.index.get_level_values(-1).to_timestamp().astype("datetime64[ns]")
            )
        elif isinstance(x_df.index.levels[-1], pd.DatetimeIndex):
            x_df["date_sequence"] = Z.index.get_level_values(-1)
        elif not isinstance(x_df.index.levels[-1], pd.DatetimeIndex):
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
