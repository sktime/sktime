#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract holiday features from datetime index."""

__author__ = ["mloning", "VyomkeshVyas", "RobKuebler"]
__all__ = ["HolidayFeatures"]

import datetime
import re
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn


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
        Days in overlapping windows of different holidays are labeled with
        both names, e.g., "Christmas, New Year".
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
        Keep original columns in X passed to ``.transform()``.
    return_offsets : bool, default=False
        If True, label holidays and their window days with the holiday name and
        the signed offset in days, e.g., "Christmas-2", "Christmas+0", "Easter+1".
        Every date gets a single label: in overlapping windows, the nearest holiday
        wins, and on a tie the upcoming one. This differs from the default mode,
        where overlapping windows get combined labels. Names joined by "; " in
        the calendar are treated as separate holidays. Names are used as given,
        e.g., including "(observed)". To avoid this, pass ``observed=False`` to
        a ``holidays`` calendar or use a dict. Dummy columns are ordered by
        holiday, then by offset.
    return_distances : bool, default=False
        If True, add a float column "<name>_distance" for each holiday in
        ``holiday_windows``, with the signed distance in days to the nearest
        occurrence of that holiday, e.g., -2 two days before, 0 on the holiday.
        Outside the window, the value is NaN; impute it for models that cannot
        handle missing values. Overlapping windows of different holidays do not
        interact, bridge days are ignored. Names joined by "; " are only split
        with ``return_offsets=True``.


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

    >>> from sktime.transformations.holiday import HolidayFeatures
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
        # packaging info
        # --------------
        "authors": ["mloning", "VyomkeshVyas"],
        "maintainers": "VyomkeshVyas",
        "python_dependencies": ["holidays"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "capability:multivariate": True,
        "capability:missing_values": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "X-y-must-have-same-index": False,
        "fit_is_empty": True,
        "requires_y": False,
        "enforce_index_type": [pd.DatetimeIndex, pd.PeriodIndex],
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
        "tests:skip_by_name": [
            "test_categorical_X_passes",
            "test_categorical_y_raises_error",
        ],  # these tests use RangeIndex data which is not supported
    }

    def __init__(
        self,
        calendar: dict[date, str],
        holiday_windows: dict[str, tuple] = None,
        include_bridge_days: bool = False,
        include_weekend: bool = False,
        return_dummies: bool = True,
        return_categorical: bool = False,
        return_indicator: bool = False,
        keep_original_columns: bool = False,
        return_offsets: bool = False,
        return_distances: bool = False,
    ) -> None:
        self.calendar = calendar
        self.holiday_windows = holiday_windows
        self.include_bridge_days = include_bridge_days
        self.include_weekend = include_weekend
        self.return_categorical = return_categorical
        self.return_dummies = return_dummies
        self.return_indicator = return_indicator
        self.keep_original_columns = keep_original_columns
        self.return_offsets = return_offsets
        self.return_distances = return_distances
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
        if isinstance(X.index, pd.PeriodIndex):
            original_index = X.index
            X_temp = X.copy()
            X_temp.index = X_temp.index.to_timestamp()

            result = self._transform(X_temp, y)

            result.index = original_index
            return result

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
            return_offsets=self.return_offsets,
            return_distances=self.return_distances,
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
            return_offsets=self.return_offsets,
            return_distances=self.return_distances,
            warning_instance=self,
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
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
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
            {
                "calendar": {date(2022, 5, 17): "Regional Holiday"},
                "include_weekend": True,
                "return_dummies": False,
                "return_categorical": True,
                "keep_original_columns": True,
            },
            {
                "calendar": {date(2022, 5, 17): "Regional Holiday"},
                "holiday_windows": {"Regional Holiday": (2, 1)},
                "include_bridge_days": True,
                "return_indicator": True,
                "return_offsets": True,
                "return_distances": True,
            },
        ]
        return params


def _generate_holidays(
    index: pd.DatetimeIndex,
    calendar: dict[date, str],
    holiday_windows: dict = None,
    include_bridge_days: bool = False,
    include_weekend: bool = False,
    return_dummies: bool = True,
    return_categorical: bool = False,
    return_indicator: bool = False,
    return_offsets: bool = False,
    return_distances: bool = False,
    warning_instance: HolidayFeatures = None,
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
    return_offsets : bool, default=False
        Whether or not to label holidays and window days with the holiday name
        and the signed offset in days, e.g., "Christmas-2".
    return_distances : bool, default=False
        Whether or not to add the signed distance in days to each holiday in
        ``holiday_windows``, NaN outside the window.
    warning_instance : HolidayFeatures, default=None
        Instance of HolidayFeatures to raise warnings.

    Returns
    -------
    pd.DataFrame
        Dataframe with index given by input ``index`` and holiday columns.

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

    # Holidays just outside the time index can still affect it through
    # their windows or bridge days, so we look them up in an extended range.
    windows = holiday_windows.values() if holiday_windows is not None else []
    pad = max([max(window) for window in windows], default=0)
    if include_bridge_days:
        pad = max(pad, 1)
    pad = datetime.timedelta(days=pad)
    lookup_dates = []
    if len(dates) > 0:
        lookup_dates = pd.date_range(dates[0] - pad, dates[-1] + pad, freq="D").date

    # We check each date for membership instead of iterating over the calendar,
    # since HolidayBase objects only populate years on lookup.
    filtered_dates = [dte for dte in lookup_dates if dte in calendar]
    if include_weekend:
        for dte in dates:
            if dte.weekday() in [5, 6]:
                holidays_by_name["Weekend"].append(dte)

    for dte in filtered_dates:
        # In offset mode, several holidays on one date are separate holidays.
        names = calendar[dte].split("; ") if return_offsets else [calendar[dte]]
        if not (include_weekend and dte.weekday() in [5, 6]):
            for name in names:
                holidays_by_name[name].append(dte)

    for name in holiday_windows or {}:
        if name not in holidays_by_name:
            warn(
                f"Holiday '{name}' not found in calendar. Skipping.",
                obj=warning_instance,
                stacklevel=2,
            )

    # For each holiday, map every date in its windows to the offset of the
    # nearest occurrence, on a tie the upcoming one (negative offset).
    def rank(days):
        return abs(days), days > 0

    nearest = defaultdict(dict)
    for name, holiday_dates in holidays_by_name.items():
        before, after = (holiday_windows or {}).get(name, (0, 0))
        offsets = nearest[name]
        for dte in holiday_dates:
            for days in range(-before, after + 1):
                date_window = dte + datetime.timedelta(days=days)
                offsets[date_window] = min(
                    offsets.get(date_window, days), days, key=rank
                )

    # Invert dictionary so that we can later map holidays to dates in the time
    # index. By default, dates in overlapping windows get all holiday names.
    # With offsets, the nearest holiday wins, as ranked above, then the first found.
    holidays_by_date = {}
    for name, offsets in nearest.items():
        for dte, days in offsets.items():
            if not return_offsets:
                holidays_by_date.setdefault(dte, []).append(name)
            elif dte not in holidays_by_date or rank(days) < holidays_by_date[dte][0]:
                label = name if name == "Weekend" else f"{name}{days:+d}"
                holidays_by_date[dte] = (rank(days), label)
    if return_offsets:
        holidays_by_date = {
            dte: [label] for dte, (_, label) in holidays_by_date.items()
        }

    if include_bridge_days:
        # Iterate over holidays.
        for name, holiday_dates in holidays_by_name.items():
            # For each holiday, iterate over all dates.
            for dte in holiday_dates:
                # Monday is a bridge day for a Tuesday holiday, Friday for a
                # Thursday holiday. Existing holidays are not overwritten.
                offset = {1: -1, 3: 1}.get(dte.weekday())
                if offset is not None:
                    bridge_day = dte + datetime.timedelta(days=offset)
                    label = f"{name}{offset:+d}" if return_offsets else name
                    holidays_by_date.setdefault(bridge_day, [label])

    # Generate categorical variable.
    labels_by_date = {dte: ", ".join(names) for dte, names in holidays_by_date.items()}
    index_dates = pd.Series(index.date, index=index)
    labels = index_dates.map(labels_by_date).fillna(no_holiday_value)
    # Order offset labels by holiday, then by offset, e.g., "Christmas-2" first.
    categories = sorted(labels.unique(), key=_offset_key if return_offsets else None)
    holidays = labels.astype(pd.CategoricalDtype(categories)).to_frame(
        name=categorical_column
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

    # Generate signed distances to each holiday in holiday_windows.
    if return_distances:
        for name in holiday_windows:
            distances = index_dates.map(nearest.get(name, {})).astype(float)
            holidays[f"{name}_distance"] = distances

    # Remove categorical variable if not requested.
    if not return_categorical:
        holidays = holidays.drop(columns=categorical_column)

    return holidays


def _offset_key(label: str):
    """Sort key splitting an offset label like "Christmas-2" into its parts."""
    match = re.fullmatch(r"(.*)([+-]\d+)", label)
    return (match[1], int(match[2])) if match else (label, 0)


def _check_params(
    index: pd.DatetimeIndex,
    calendar: dict[date, str],
    holiday_windows: dict[str, tuple],
    include_bridge_days: bool,
    include_weekend: bool,
    return_dummies: bool,
    return_categorical: bool,
    return_indicator: bool,
    keep_original_columns: bool,
    return_offsets: bool = False,
    return_distances: bool = False,
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
    return_offsets : bool
    return_distances : bool
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
    if not isinstance(return_offsets, bool):
        raise ValueError(
            f"`return_offsets` must be a boolean, but found: {return_offsets}"
        )
    if not isinstance(return_distances, bool):
        raise ValueError(
            f"`return_distances` must be a boolean, but found: {return_distances}"
        )
    if return_distances and not holiday_windows:
        raise ValueError("`return_distances=True` requires `holiday_windows`.")
    if not (
        return_dummies or return_categorical or return_indicator or return_distances
    ):
        raise ValueError(
            "One of `return_dummies`, `return_categorical`, `return_indicator` "
            "and `return_distances` must be set to True."
        )
    if not isinstance(calendar, HolidayBase) and isinstance(calendar, dict):
        _check_calendar(calendar)

    if holiday_windows is not None:
        _check_holiday_windows(holiday_windows)


def _check_holiday_windows(holiday_windows: dict[str, tuple]):
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
            if not (isinstance(days, int) and days >= 0):
                raise ValueError(
                    "days in `holiday_windows` must all be non-negative, "
                    f"but found: {holiday}: {window}"
                )


def _check_calendar(calendar: dict[date, str]):
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
