# -*- coding: utf-8 -*-

"""Holiday functionality."""

import datetime
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from holidays import HolidayBase
from sktime.transformations.base import _SeriesToSeriesTransformer
from holidays import CountryHoliday


class HolidayFeatures(_SeriesToSeriesTransformer):
    """Holiday features.

    Parameters
    ----------
    calendar : HolidayBase object
        Calendar object from holidays package [1]_.
    holiday_windows : Dict[str, tuple], default=None
        Dictionary for specifying a window of days around holidays, with keys
        being holiday names and values being (n_days_before, n_days_after) tuples.
    include_bridge_days: bool, default=False
        If True, include bridge days. Bridge days include Monday if a holiday
        is on Tuesday and Friday if a holiday is on Thursday.
    return_dummies : bool, default=True
        Whether or not to return a dummy variable for each holiday.
    return_categorical : bool, default=False
        Whether or not to return a categorical variable with holidays
        beings categories.
    return_indicator : bool, default=False
        Whether or not to return an indicator variable equal to 1 if a time
        point is a holiday or not.

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
        "enforce_index_type": "pd.DatetimeIndex",
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
    }

    def __init__(
        self,
        calendar: HolidayBase,
        holiday_windows: Dict[str, tuple] = None,
        include_bridge_days: bool = False,
        return_dummies: bool = True,
        return_categorical: bool = False,
        return_indicator: bool = False,
    ) -> None:
        self.calendar = calendar
        self.holiday_windows = holiday_windows
        self.include_bridge_days = include_bridge_days
        self.return_categorical = return_categorical
        self.return_dummies = return_dummies
        self.return_indicator = return_indicator
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
        holidays = _generate_holidays(
            X.index,
            calendar=self.calendar,
            holiday_windows=self.holiday_windows,
            include_bridge_days=self.include_bridge_days,
            return_categorical=self.return_categorical,
            return_dummies=self.return_dummies,
            return_indicator=self.return_indicator,
        )
        return pd.concat([X, holidays], axis=1)

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
        return {"calendar": CountryHoliday(country="FR")}


def _generate_holidays(
    index: pd.DatetimeIndex,
    calendar: HolidayBase,
    holiday_windows: dict = None,
    include_bridge_days: bool = False,
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
    calendar : HolidayBase object
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

    # Input checks.
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            f"Time index must be of type pd.DatetimeIndex, but found: {type(index)}"
        )
    if not isinstance(calendar, HolidayBase):
        raise ValueError(
            "calendar must be of type HolidayBase from the `holidays` package, "
            " but found: {type(index)}."
        )
    if not isinstance(return_dummies, bool):
        raise ValueError(
            f"`return_dummies` must be a boolean, but found: {return_dummies}"
        )
    if not isinstance(return_categorical, bool):
        raise ValueError(
            f"`return_categorical` must be a boolean, but found: {return_dummies}"
        )
    if not isinstance(return_indicator, bool):
        raise ValueError(
            f"`return_indicator` must be a boolean, but found: {return_dummies}"
        )
    if not isinstance(include_bridge_days, bool):
        raise ValueError(
            f"`include_bridge_days` must be a boolean, but found: {return_dummies}"
        )
    if not (return_dummies or return_categorical or return_indicator):
        raise ValueError(
            "One of `return_dummies`, `return_categorical` and `return_indicator` "
            "must be set to True."
        )
    if holiday_windows is not None:
        _check_holiday_windows(holiday_windows)

    # Define variable names and fixed values.
    categorical_column = "holiday"
    indicator_column = "is_holiday"
    no_holiday_value = "no_holiday"

    # Get holiday dictionary by name with
    # values being a list, since we may observe
    # holidays over multiple years.
    dates = np.unique(index.date)
    holidays_by_name = defaultdict(list)
    for date in filter(lambda date: date in calendar, dates):
        name = calendar[date]
        holidays_by_name[name].append(date)

    # Invert dictionary so that we can later map holidays to
    # dates in the time index.
    holidays_by_date = {}
    for name, dates in holidays_by_name.items():
        for date in dates:
            holidays_by_date[date] = name

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
            for date in dates:

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
                    date_before = date - datetime.timedelta(days=days)
                    if date_before not in holidays_by_date:
                        holidays_by_date[date_before] = name
                    else:
                        raise ValueError(msg)

                for days in range(1, after + 1):
                    date_after = date + datetime.timedelta(days=days)
                    if date_after not in holidays_by_date:
                        holidays_by_date[date_after] = name
                    else:
                        raise ValueError(msg)

    if include_bridge_days:
        # Iterate over holidays.
        for name, dates in holidays_by_name.items():

            # For each holiday, iterate over all dates.
            for date in dates:

                # Get the weekday of the holiday.
                weekday = date.weekday()

                # If the holiday is on Tuesday, we add Monday as a bridge day.
                if weekday == 1:
                    bridge_day = date - datetime.timedelta(days=1)

                    # We only add bridge days if they are not holidays already.
                    if bridge_day not in holidays_by_date:
                        holidays_by_date[bridge_day] = name

                # If the holiday is on Thursday, we add Friday as a bridge day.
                if weekday == 3:
                    bridge_day = date + datetime.timedelta(days=1)

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


def _check_holiday_windows(holiday_windows: Dict[str, tuple]):
    """Check holiday windows.

    Parameters
    ----------
    holiday_windows : Dict[str, tuple]
        Dictionary with keys being holiday names and values being
        (n_days_before, n_days_after) tuples.

    Returns
    -------
    Dict[str, tuple]
        Checked holiday windows.
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

    return holiday_windows
