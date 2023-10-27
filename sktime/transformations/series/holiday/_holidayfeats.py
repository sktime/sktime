#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract holiday features from datetime index."""

__author__ = ["mloning", "VyomkeshVyas"]
__all__ = ["HolidayFeatures"]

import datetime
from collections import defaultdict
from datetime import date
from typing import Dict

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies


class HolidayFeatures(BaseTransformer):
    """Holiday features extraction.

    HolidayFeatures uses a dictionary of holidays (which could be a custom made dict
    or imported as HolidayBase object from holidays package) to extract
    holiday features from a datetime index.

    Parameters
    ----------
    calendar : HolidayBase object or Dict[date, str]
        Calendar object from holidays package [1]_.
    holiday_windows : Dict[str, tuple], default=None
        Dictionary for specifying a window of days around holidays, with keys
        being holiday names and values being (n_days_before, n_days_after) tuples.
    include_bridge_days: bool, default=False
        If True, include bridge days. Bridge days include Monday if a holiday
        is on Tuesday and Friday if a holiday is on Thursday.
    include_weekend: bool, default=False
        If True, include weekends as holidays.
    return_dummies : bool, default=True
        Whether or not to return a dummy variable for each holiday.
    return_categorical : bool, default=False
        Whether or not to return a categorical variable with holidays
        beings categories.
    return_indicator : bool, default=False
        Whether or not to return an indicator variable equal to 1 if a time
        point is a holiday or not.
    keep_original_columns : bool, default=False
        Keep original columns in X passed to `.transform()`.

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> import pandas as pd  # doctest: +SKIP
    >>> from datetime import date  # doctest: +SKIP
    >>> from holidays import country_holidays, financial_holidays  # doctest: +SKIP
    >>> values = np.random.normal(size=365)  # doctest: +SKIP
    >>> index = pd.date_range("2000-01-01", periods=365, freq="D")  # doctest: +SKIP
    >>> X = pd.DataFrame(values, index=index)  # doctest: +SKIP

    Returns country holiday features with custom holiday windows

    >>> transformer = HolidayFeatures(
    ...    calendar=country_holidays(country="FR"),
    ...    return_categorical=True,
    ...    holiday_windows={"Noël": (1, 3), "Jour de l'an": (1, 0)})  # doctest: +SKIP
    >>> yt = transformer.fit_transform(X)  # doctest: +SKIP

    Returns financial holiday features

    >>> transformer = HolidayFeatures(
    ...    calendar=financial_holidays(market="NYSE"),
    ...    return_categorical=True,
    ...    include_weekend=True)  # doctest: +SKIP
    >>> yt = transformer.fit_transform(X)  # doctest: +SKIP

    Returns custom made holiday features

    >>> transformer = HolidayFeatures(
    ...    calendar={date(2000,1,14): "Regional Holiday",
    ...              date(2000, 1, 26): "Regional Holiday"},
    ...    return_categorical=True)  # doctest: +SKIP
    >>> yt = transformer.fit_transform(X)  # doctest: +SKIP

    References
    ----------
    .. [1] https://pypi.org/project/holidays/
    """

    _required_parameters = ["calendar"]
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "univariate-only": False,
        "handles-missing-data": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "X-y-must-have-same-index": False,
        "fit_is_empty": True,
        "requires_y": False,
        "enforce_index_type": [pd.DatetimeIndex],
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        "python_dependencies": ["holidays"],
    }

    def __init__(
        self,
        calendar: Dict[date, str],
        holiday_windows: Dict[str, tuple] = None,
        include_bridge_days: bool = False,
        include_weekend: bool = False,
        return_dummies: bool = True,
        return_categorical: bool = False,
        return_indicator: bool = False,
        keep_original_columns: bool = False,
    ) -> None:
        self.calendar = calendar
        self.holiday_windows = holiday_windows
        self.include_bridge_days = include_bridge_days
        self.include_weekend = include_weekend
        self.return_categorical = return_categorical
        self.return_dummies = return_dummies
        self.return_indicator = return_indicator
        self.keep_original_columns = keep_original_columns
        super().__init__()

    def _transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : Series
            Time series
        y : Series, default=None
            Time series

        Returns
        -------
        Series
            Input series with generated holiday features.
        """
        _check_params(
            X.index,
            calendar=self.calendar,
            holiday_windows=self.holiday_windows,
            include_bridge_days=self.include_bridge_days,
            include_weekend=self.include_weekend,
            return_categorical=self.return_categorical,
            return_dummies=self.return_dummies,
            return_indicator=self.return_indicator,
            keep_original_columns=self.keep_original_columns,
        )

        holidays = _generate_holidays(
            X.index,
            calendar=self.calendar,
            holiday_windows=self.holiday_windows,
            include_bridge_days=self.include_bridge_days,
            include_weekend=self.include_weekend,
            return_categorical=self.return_categorical,
            return_dummies=self.return_dummies,
            return_indicator=self.return_indicator,
        )

        if self.keep_original_columns:
            return pd.concat([X, holidays], axis=1, copy=True)
        else:
            return holidays

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from datetime import date

        if _check_soft_dependencies("holidays", severity="none"):
            from holidays import country_holidays, financial_holidays

            params = [
                {
                    "calendar": dict(country_holidays(country="GB")),
                    "include_weekend": True,
                    "return_categorical": True,
                    "return_dummies": True,
                },
                {
                    "calendar": dict(financial_holidays(market="NYSE")),
                    "return_indicator": True,
                    "include_bridge_days": True,
                    "return_dummies": False,
                },
            ]
        else:
            params = []

        params += [
            {
                "calendar": {date(2022, 5, 15): "Regional Holiday"},
                "return_indicator": True,
            },
        ]
        return params


def _generate_holidays(
    index: pd.DatetimeIndex,
    calendar: Dict[date, str],
    holiday_windows: dict = None,
    include_bridge_days: bool = False,
    include_weekend: bool = False,
    return_dummies: bool = True,
    return_categorical: bool = False,
    return_indicator: bool = False,
) -> pd.DataFrame:
    """Generate holidays.

    This looks up holidays in the calendar for the given time index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index with time points for which to generate holidays.
    calendar : HolidayBase object or Dict[date, str]
        Calendar object from holidays package [1]_.
    include_bridge_days: bool, default=False
        If True, include bridge days. Bridge days include Monday if a holiday
        is on Tuesday and Friday if a holiday is on Thursday.
    holiday_windows : Dict[str, tuple], default=None
        Dictionary for specifying a window of days around holidays, with keys
        being holiday names and values being (n_days_before, n_days_after) tuples.
    return_dummies : bool, default=True
        Whether or not to return a dummy variable for each holiday.
    return_categorical : bool, default=False
        Whether or not to return a categorical variable with holidays
        beings categories.
    return_indicator : bool, default=False
        Whether or not to return an indicator variable equal to 1 if a time
        point is a holiday or not.

    Returns
    -------
    pd.DataFrame
        Dataframe with index given by input `index` and holiday columns.

    References
    ----------
    .. [1] https://pypi.org/project/holidays/
    """
    # Note that we currently handle bridge days and windows around holidays
    # as part of the holiday generation, it may be better placed in
    # a separate calendar module.

    # Define variable names and fixed values.
    categorical_column = "holiday"
    indicator_column = "is_holiday"
    no_holiday_value = "no_holiday"

    # Get holiday dictionary by name with
    # values being a list, since we may observe
    # holidays over multiple years.
    dates = np.unique(index.date)
    holidays_by_name = defaultdict(list)

    filtered_dates = [dte for dte in dates if dte in calendar]
    weekends = []
    if include_weekend:
        for dte in dates:
            if dte.weekday() in [5, 6]:
                holidays_by_name["Weekend"].append(dte)
        weekends = holidays_by_name["Weekend"]

    for dte in filtered_dates:
        name = calendar[dte]
        if dte not in weekends:
            holidays_by_name[name].append(dte)

    # Invert dictionary so that we can later map holidays to
    # dates in the time index.
    holidays_by_date = {}
    for name, dates in holidays_by_name.items():
        for dte in dates:
            holidays_by_date[dte] = name

    # Add window around holidays.
    if holiday_windows is not None:
        # Iterate over holidays.
        for name, window in holiday_windows.items():
            # First, we look up the dates of the holiday.
            if name in holidays_by_name:
                dates = holidays_by_name[name]
            else:
                raise ValueError(f"holiday: {name} not found in calendar.")

            # For each holiday, we iterate over dates.
            for dte in dates:
                # We then get the number of days before and after
                # the holiday.
                before, after = window

                # Finally, we add all days within the window to the
                # holiday, making sure that we do not overwrite
                # already existing holidays.
                msg = (
                    "Holiday already exists. Please make sure holiday windows "
                    "are not overlapping with other holidays."
                )

                for days in range(1, before + 1):
                    date_before = dte - datetime.timedelta(days=days)
                    if date_before not in holidays_by_date:
                        holidays_by_date[date_before] = name
                    else:
                        raise ValueError(msg)

                for days in range(1, after + 1):
                    date_after = dte + datetime.timedelta(days=days)
                    if date_after not in holidays_by_date:
                        holidays_by_date[date_after] = name
                    else:
                        raise ValueError(msg)

    if include_bridge_days:
        # Iterate over holidays.
        for name, dates in holidays_by_name.items():
            # For each holiday, iterate over all dates.
            for dte in dates:
                # Get the weekday of the holiday.
                weekday = dte.weekday()

                # If the holiday is on Tuesday, we add Monday as a bridge day.
                if weekday == 1:
                    bridge_day = dte - datetime.timedelta(days=1)

                    # We only add bridge days if they are not holidays already.
                    if bridge_day not in holidays_by_date:
                        holidays_by_date[bridge_day] = name

                # If the holiday is on Thursday, we add Friday as a bridge day.
                if weekday == 3:
                    bridge_day = dte + datetime.timedelta(days=1)

                    # We only add bridge days if they are not holidays already.
                    if bridge_day not in holidays_by_date:
                        holidays_by_date[bridge_day] = name

    # Generate categorical variable.
    holidays = (
        index.to_series()
        .dt.date.map(holidays_by_date)
        .fillna(no_holiday_value)
        .astype("category")
        .to_frame(name=categorical_column)
        .set_index(index)
    )

    # Generate dummies.
    if return_dummies:
        dummies = pd.get_dummies(
            holidays,
            columns=[categorical_column],
            prefix="",
            prefix_sep="",
            dtype=int,
        )
        if no_holiday_value in dummies.columns:
            dummies = dummies.drop(columns=no_holiday_value)
        holidays = pd.concat([holidays, dummies], axis=1)

    # Generate indicator.
    if return_indicator:
        holidays[indicator_column] = (
            holidays[categorical_column] != no_holiday_value
        ).astype(int)

    # Remove categorical variable if not requested.
    if not return_categorical:
        holidays = holidays.drop(columns=categorical_column)

    return holidays


def _check_params(
    index: pd.DatetimeIndex,
    calendar: Dict[date, str],
    holiday_windows: Dict[str, tuple],
    include_bridge_days: bool,
    include_weekend: bool,
    return_dummies: bool,
    return_categorical: bool,
    return_indicator: bool,
    keep_original_columns: bool,
):
    """Check input params.

    Parameters
    ----------
    index : pd.DatetimeIndex
    calendar : Dict[date, str],
    include_bridge_days: bool
    include_weekend: bool
    holiday_windows : Dict[str, tuple]
    return_dummies : bool
    return_categorical : bool
    return_indicator : bool
    keep_original_columns : bool
    """
    from holidays import HolidayBase

    # Input checks.
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            f"Time index must be of type pd.DatetimeIndex, but found: {type(index)}"
        )
    if not isinstance(calendar, HolidayBase) and not isinstance(calendar, dict):
        raise ValueError(
            f"calendar must be either of type HolidayBase from the `holidays` package, "
            f" or a dict, but found: {type(calendar)}."
        )
    if not isinstance(return_dummies, bool):
        raise ValueError(
            f"`return_dummies` must be a boolean, but found: {return_dummies}"
        )
    if not isinstance(return_categorical, bool):
        raise ValueError(
            f"`return_categorical` must be a boolean, but found: {return_categorical}"
        )
    if not isinstance(return_indicator, bool):
        raise ValueError(
            f"`return_indicator` must be a boolean, but found: {return_indicator}"
        )
    if not isinstance(include_bridge_days, bool):
        raise ValueError(
            f"`include_bridge_days` must be a boolean, but found: {include_bridge_days}"
        )
    if not isinstance(include_weekend, bool):
        raise ValueError(
            f"`include_weekend` must be a boolean, but found: {include_weekend}"
        )
    if not isinstance(keep_original_columns, bool):
        raise ValueError(
            f"`keep_original_columns` must be boolean,"
            f"but found; {keep_original_columns}"
        )
    if not (return_dummies or return_categorical or return_indicator):
        raise ValueError(
            "One of `return_dummies`, `return_categorical` and `return_indicator` "
            "must be set to True."
        )
    if not isinstance(calendar, HolidayBase) and isinstance(calendar, dict):
        _check_calendar(calendar)

    if holiday_windows is not None:
        _check_holiday_windows(holiday_windows)


def _check_holiday_windows(holiday_windows: Dict[str, tuple]):
    """Check holiday windows.

    Parameters
    ----------
    holiday_windows : Dict[str, tuple]
        Dictionary with keys being holiday names and values being
        (n_days_before, n_days_after) tuples.

    """
    if not isinstance(holiday_windows, dict):
        raise ValueError(
            "`holiday_windows` must be a dictionary, "
            f"but found: {type(holiday_windows)}"
        )
    for holiday, window in holiday_windows.items():
        if not (
            isinstance(holiday, str) and isinstance(window, tuple) and len(window) == 2
        ):
            raise ValueError(
                "`holiday_windows` must be a dictionary, with keys being strings "
                "and values tuples of length 2"
            )
        for days in window:
            if not isinstance(days, int) and days >= 0:
                raise ValueError(
                    "days in `holiday_windows` must all be non-negative, "
                    f"but found: {holiday}: {window}"
                )


def _check_calendar(calendar: Dict[date, str]):
    """Check calendar param.

    Parameters
    ----------
    calendar : Dict[date, str]
        Dictionary with keys being holiday dates and values being
        holiday names.

    """
    for dte, name in calendar.items():
        if not (isinstance(dte, date) and isinstance(name, str)):
            raise ValueError(
                "`calendar` must be a dictionary, with keys being date "
                "and value being name of holiday."
            )
