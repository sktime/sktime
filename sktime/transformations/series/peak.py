# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract peak/working hour features from datetimeindex."""

__author__ = ["ali-parizad"]
__all__ = ["PeakTimeFeature"]


import pandas as pd

from sktime.transformations.base import BaseTransformer


class PeakTimeFeature(BaseTransformer):
    """PeakTime feature extraction for use in e.g. tree based models.

    PeakTimeFeature uses a datetime index column and generates peak/working features
    for e.g. peak hours, peak weak, peak month, working hours, etc. It works based on
    input intervals, start time, end time e.g., peak_hour_start=[6], peak_hour_end=[9]

    Parameters
    ----------
    ts_freq : str, default= None
        Restricts selection of items to those with a frequency lower than
        the frequency of the time series given by ts_freq.
        E.g., if daily data is provided and ts_freq = ("D"), it does not make
        sense to derive PeakTimeFeature with higher frequency like hourly features. So,
        the outpul must exclude is_peak_hour and will be is_peak_day, is_peak_week,
        is_peak_month, is_peak_quarter, is_peak_year.
        Only supports the following frequencies:
        {"Y": year, "Q": quarter, "M": month, "W": week, "D": day, "H": hour}
    peak_hour_start : list, default= None
        start interval of peak hour(s) in a list where the first argument determines the
        first start hour, the second one determines the second start hour and so on.
        [peak_hour_start1, peak_hour_start2, peak_hour_start3, ...]
    peak_hour_end : list, default= None
        end interval of peak hour(s) in a list where the first argument determines the
        first end hour, the second one determines the second end hour and so on.
        [peak_hour_end1, peak_hour_end2, peak_hour_end3, ...]
    peak_day_start : list, default= None
        start interval of peak day(s) in a list where the first argument determines the
        first start day, the second one determines the second start day and so on.
        [peak_day_start1, peak_day_start2, peak_day_start3, ...]
    peak_day_end : list, default= None
        end interval of peak day(s) in a list where the first argument determines the
        first end day, the second one determines the second end day and so on.
        [peak_day_end1, peak_day_end2, peak_day_end3, ...]
    peak_week_start : list, default= None
        start interval of peak week(s) in a list where the first argument determines
        the first start week, the second one determines the second start week and so on.
        [peak_week_start1, peak_week_start2, peak_week_start3, ...]
    peak_week_end : list, default= None
        end interval of peak week(s) in a list where the first argument determines the
        first end week, the second one determines the second end week and so on.
        [peak_week_end1, peak_week_end2, peak_week_end3, ...]
    peak_month_start : list, default= None
        start interval of peak month(s) in a list where the first argument determines
        the first start month, the second one determines the second start month
        and so on. [peak_month_start1, peak_month_start2, peak_month_start3, ...]
    peak_month_end : list, default= None
        end interval of peak month(s) in a list where the first argument determines the
        first end month, the second one determines the second end month and so on.
        [peak_month_end1, peak_month_end2, peak_month_end3, ...]
    peak_quarter_start : list, default= None
        start interval of peak quarter(s) in a list where the first argument determines
        first start quarter, the second one determines the second start quarter
        and so on. [peak_quarter_start1, peak_quarter_start2, peak_quarter_start3, ...]
    peak_quarter_end : list, default= None
        end interval of peak quarter(s) in a list where the first argument determines
        first end quarter, the second one determines the second end quarter and so on.
        [peak_quarter_end1, peak_quarter_end2, peak_quarter_end3, ...]
    peak_year_start : list, default= None
        start interval of peak year(s) in a list where the first argument determines
        first start year, the second one determines the second start year and so on.
        [peak_year_start1, peak_year_start2, peak_year_start3, ...]
    peak_year_end : list, default= None
        end interval of peak year(s) in a list where the first argument determines
        first end year, the second one determines the second end year and so on.
        [peak_year_end1, peak_year_end2, peak_year_end3, ...]
    working_hour_start : list, default= None
        start interval of working hour(s) in a list where the first argument determines
        first start working hour, the second one determines the second start working
        hour and so on.
        e.g., [working_hour_start1, working_hour_start2, working_hour_start3, ...]
    working_hour_end : list, default= None
        end interval of working hour(s) in a list where the first argument determines
        first end working hour, the second one determines the second end working hour
        and so on. [working_hour_end1, working_hour_end2, working_hour_end3, ...]
    keep_original_columns :  boolean, optional, default=False
        If True, keep original columns in main dataframe (X) passed to ``.transform()``.
    keep_original_peaktime_data_columns: boolean, optional, default=False
        If True, keep original peaktime_data dataframe columns including all separate
        peak/working columns, e.g., peak_hour_1, peak_hour_2, peak_week_1,
        peak_week_2, ...

    Notes
    -----
    * Descriptions and offsets for calendar features are as follows:
    hour:
        The hour of the day with 00:00:00 = 0, 23:00:00 = 23.
    day_of_week:
        The day of the week with Monday=0, Sunday=6.
    week_of_year:
        The week of the year (ISO calendar), start = 1, end = 52 or 53
    month_of_year:
        The month as January=1, December=12.
    quarter:
        The quarter of the year as:
        January, February, March = 1
        April, May, June, = 2
        July, August, September = 3
        October, November, December = 4
    year:
        The year of the datetime.

    * Descriptions for different intervals:

    1- Example for one peak interval: peak_hour_start=[6], peak_hour_end=[9] means we
    have just one peak hour interval where peak starts at 6am and peak ends at 9am.

    2- Example for two peak intervals: peak_hour_start=[6, 16], peak_hour_end=[9, 20]
    means we have two peak hour intervals where the first peak starts at 6 am
    and ends at 9 am. The second peak starts at 16 am and ends at 20. we can
    have more than two intervals.

    3- Example for one working interval: working_hour_start=[8], working_hour_end=[16]
    means we have just one working hour interval where work starts at 8 am and
    work ends at 16.

    4- Example for two working intervals: working_hour_start=[8, 15],
    working_hour_end=[12, 19] means we have two working hour intervals where the
    first starts at 8 am and ends at 15. The second  starts at 15 am and ends at 19.
    we can have more than two intervals.

    Examples
    --------
    >>> from sktime.transformations.series.peak import PeakTimeFeature  # doctest: +SKIP
    >>> from sktime.datasets import   # doctest: +SKIP
    >>> y = load_solar()  # doctest: +SKIP
    >>> y = y.tz_localize(None)  # doctest: +SKIP
    >>> y = y.asfreq("H")  # doctest: +SKIP

    Example 1: one interval for peak hour and working hour.
    (based on one start/end interval)

    Returns columns is_peak_hour, is_working_hour

    >>> transformer = PeakTimeFeature(ts_freq="H",
    ... peak_hour_start=[6], peak_hour_end=[9],
    ... working_hour_start=[8], working_hour_end=[16]
    ... )  # doctest: +SKIP
    >>> y_hat_peak = transformer.fit_transform(y)  # doctest: +SKIP

    Example 2: two intervals for peak hour and  working hour.
    (based on two start/end intervals)

    Returns columns is_peak_hour, is_working_hour

    >>> transformer = PeakTimeFeature(ts_freq="H",
    ... peak_hour_start=[6, 16], peak_hour_end=[9, 20],
    ... working_hour_start=[8, 15], working_hour_end=[12, 19]
    ... )  # doctest: +SKIP
    >>> y_hat_peak = transformer.fit_transform(y)  # doctest: +SKIP

    Example 3: We may have peak for different seasonality
    Here is an example for peak hour, peak day, peak week, peak month for
    two intervals (based on two start/end intervals)

    Returns columns is_peak_hour, is_peak_day, is_peak_week, is_peak_month

    >>> transformer = PeakTimeFeature(ts_freq="H",
    ... peak_hour_start=[6, 16], peak_hour_end=[9, 20],
    ... peak_day_start=[1, 2], peak_day_end=[2, 3],
    ... peak_week_start=[35, 45], peak_week_end=[40, 52],
    ... peak_month_start=[1, 7], peak_month_end=[6, 12]
    ... )  # doctest: +SKIP
    >>> y_hat_peak = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ali-parizad"],
        "maintainers": ["ali-parizad"],
        "python_dependencies": "pandas>=1.2.0",  # from DateTimeProperties
        # estimator type
        # --------------
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
    }

    def __init__(
        self,
        ts_freq=None,
        peak_hour_start=None,
        peak_hour_end=None,
        peak_day_start=None,
        peak_day_end=None,
        peak_week_start=None,
        peak_week_end=None,
        peak_month_start=None,
        peak_month_end=None,
        peak_quarter_start=None,
        peak_quarter_end=None,
        peak_year_start=None,
        peak_year_end=None,
        working_hour_start=None,
        working_hour_end=None,
        keep_original_columns=False,
        keep_original_peaktime_data_columns=False,
    ):
        self.ts_freq = ts_freq
        self.peak_hour_start = peak_hour_start
        self.peak_hour_end = peak_hour_end
        self.peak_day_start = peak_day_start
        self.peak_day_end = peak_day_end
        self.peak_week_start = peak_week_start
        self.peak_week_end = peak_week_end
        self.peak_month_start = peak_month_start
        self.peak_month_end = peak_month_end
        self.peak_quarter_start = peak_quarter_start
        self.peak_quarter_end = peak_quarter_end
        self.peak_year_start = peak_year_start
        self.peak_year_end = peak_year_end
        self.working_hour_start = working_hour_start
        self.working_hour_end = working_hour_end
        self.keep_original_columns = keep_original_columns
        self.keep_original_peaktime_data_columns = keep_original_peaktime_data_columns

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
        _check_inputs(self.peak_hour_start, self.peak_hour_end, "peak_hour", 0, 23)
        _check_inputs(self.peak_day_start, self.peak_day_end, "peak_day", 0, 6)
        _check_inputs(self.peak_week_start, self.peak_week_end, "peak_week", 1, 53)
        _check_inputs(self.peak_month_start, self.peak_month_end, "peak_month", 1, 12)
        _check_inputs(
            self.peak_quarter_start, self.peak_quarter_end, "peak_quarter", 1, 4
        )
        _check_inputs(self.peak_year_start, self.peak_year_end, "peak_year", 1900, 2100)
        _check_inputs(
            self.working_hour_start, self.working_hour_end, "working_hour", 0, 23
        )

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

        calendar_features = _extract_datetime_features(x_df)

        datetime_freq = _datetime_frequency_rank()

        # check time series freq
        _check_ts_freq(x_df, datetime_freq, self.ts_freq)

        df = self._extract_peaktime_features(calendar_features, datetime_freq)

        if self.keep_original_columns:
            to_concat = [X, df]
            to_concat = [df for df in to_concat if not df.empty]
            Xt = pd.concat(to_concat, axis=1, copy=True)
        else:
            Xt = df

        return Xt

    def _extract_peaktime_features(
        self,
        calendar_features,
        datetime_freq,
    ):
        """Extract peaktime features.

        Parameters
        ----------
        calendar_features : DataFrame
            DataFrame with columns (hour, day_of_week, day_of_year, week_of_year
            month_of_year, quarter, year)

        datetime_freq : str
            Frequency of main dataframe (x_df)

        Returns
        -------
        peaktime_data: pd.DataFrame including datetime features, 'is_peak_hour_',
        'is_peak_day_', 'is_peak_week_', 'is_peak_month_', 'is_peak_quarter_', '
        is_peak_year_', 'is_working_hour_'
        """
        peaktime_data = calendar_features.copy()

        # Create is_peak_ columns
        peak_config = {
            "hour": ("H", self.peak_hour_start, self.peak_hour_end, "is_peak_hour"),
            "day_of_week": ("D", self.peak_day_start, self.peak_day_end, "is_peak_day"),
            "week_of_year": (
                "W",
                self.peak_week_start,
                self.peak_week_end,
                "is_peak_week",
            ),
            "month_of_year": (
                "M",
                self.peak_month_start,
                self.peak_month_end,
                "is_peak_month",
            ),
            "quarter": (
                "Q",
                self.peak_quarter_start,
                self.peak_quarter_end,
                "is_peak_quarter",
            ),
            "year": ("Y", self.peak_year_start, self.peak_year_end, "is_peak_year"),
        }

        for freq_name, (
            freq_short,
            start_values,
            end_values,
            is_peak_col,
        ) in peak_config.items():
            if (
                start_values is not None
                and end_values is not None
                and self.ts_freq
                in datetime_freq.loc[
                    datetime_freq["rank"]
                    >= (
                        datetime_freq.loc[
                            datetime_freq["frequency"] == freq_short, "rank"
                        ].max()
                    )
                ]["frequency"].tolist()
            ):
                for i, (start, end) in enumerate(zip(start_values, end_values)):
                    peaktime_data[f"{is_peak_col}_{i + 1}"] = (
                        (peaktime_data[f"{freq_name}"] >= start)
                        & (peaktime_data[f"{freq_name}"] <= end)
                    ).astype(bool)

                peak_columns = [
                    col
                    for col in peaktime_data.columns
                    if col.startswith(f"{is_peak_col}_")
                ]
                peaktime_data[is_peak_col] = (
                    peaktime_data[peak_columns].any(axis=1).astype(int)
                )

        # Create is_working_ columns
        work_config = {
            "hour": (
                "H",
                self.working_hour_start,
                self.working_hour_end,
                "is_working_hour",
            ),
        }

        for freq_name, (
            freq_short,
            start_values,
            end_values,
            is_working_col,
        ) in work_config.items():
            if (
                start_values is not None
                and end_values is not None
                and self.ts_freq
                in datetime_freq.loc[
                    datetime_freq["rank"]
                    >= (
                        datetime_freq.loc[
                            datetime_freq["frequency"] == freq_short, "rank"
                        ].max()
                    )
                ]["frequency"].tolist()
            ):
                for i, (start, end) in enumerate(zip(start_values, end_values)):
                    peaktime_data[f"{is_working_col}_{i + 1}"] = (
                        (peaktime_data[f"{freq_name}"] >= start)
                        & (peaktime_data[f"{freq_name}"] <= end)
                    ).astype(bool)

                peak_columns = [
                    col
                    for col in peaktime_data.columns
                    if col.startswith(f"{is_working_col}_")
                ]
                peaktime_data[is_working_col] = (
                    peaktime_data[peak_columns].any(axis=1).astype(int)
                )

        if not self.keep_original_peaktime_data_columns:
            columns_to_drop = [
                col
                for col in peaktime_data.columns
                if not col.startswith("is_") or col[-1].isdigit()
            ]
            peaktime_data.drop(columns=columns_to_drop, inplace=True)

        return peaktime_data

    @classmethod
    def get_test_params(cls):
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
        params1 = {
            "peak_day_start": [1, 4],
            "peak_day_end": [2, 5],
            "keep_original_columns": True,
        }
        params2 = {
            "peak_week_start": [35, 45],
            "peak_week_end": [40, 52],
            "working_hour_start": [8, 15],
            "working_hour_end": [12, 20],
            "keep_original_columns": False,
            "keep_original_peaktime_data_columns": True,
        }
        return [params1, params2]


def _datetime_frequency_rank():
    column_names = ["date_order", "frequency", "rank"]
    date_order = [
        ["year", "Y", 0],
        ["quarter", "Q", 1],
        ["month", "M", 2],
        ["week", "W", 3],
        ["day", "D", 4],
        ["hour", "H", 5],
    ]
    datetime_freq = pd.DataFrame(date_order, columns=column_names)
    return datetime_freq


def _extract_datetime_features(x_df):
    """Extract datetime features e.g., hour, day_of_week, from datetime index.

    Parameters
    ----------
    x_df : DataFrame
        Dataframe with time index

    Returns
    -------
    calendar_features : DataFrame
         DataFrame with new columns (hour, day_of_week, day_of_year, week_of_year
         month_of_year, quarter, year)
    """
    date_sequence = x_df["date_sequence"]
    calendar_features = pd.DataFrame()
    calendar_features["hour"] = date_sequence.dt.hour
    calendar_features["day_of_week"] = date_sequence.dt.dayofweek
    calendar_features["day_of_year"] = date_sequence.dt.dayofyear
    calendar_features["week_of_year"] = date_sequence.dt.isocalendar().week
    calendar_features["month_of_year"] = date_sequence.dt.month
    calendar_features["quarter"] = date_sequence.dt.quarter
    calendar_features["year"] = date_sequence.dt.year
    return calendar_features


def _check_inputs(start_values, end_values, feature_name, start_range, end_range):
    """Check ranges of all inputs to be valid.

    It checks input intervals (start_values and end_values) to see whether it is
    in the specified ranges (start_range, end_range) or not.

    Parameters
    ----------
    start_values : int
        start time in the interval
    end_values : int
        end time in the interval
    feature_name : str
        checks the start and end ranges for all of feature names (e.g., "peak_hour")
    start_range : int
        valid start range
    end_range : int
        valid end range

    Raises
    ------
    ValueError
        Raise an error if start or end values are not in the specified
        start_range and end_range
    """
    if start_values is None or end_values is None:
        return
    for start, end in zip(start_values, end_values):
        if start is not None and (start < start_range or start > end_range):
            raise ValueError(
                f"Invalid {feature_name}_start value: {start}. It should be between"
                f" {start_range} and {end_range}."
            )
        if end is not None and (end < start_range or end > end_range):
            raise ValueError(
                f"Invalid {feature_name}_end value: {end}. It should be between"
                f" {start_range} and {end_range}."
            )
        if start > end:
            raise ValueError(
                f"Invalid {feature_name}_end value: {end}. It must be "
                f"greater than start_value (or equal): {start} "
            )


def _check_ts_freq(x_df, datetime_freq, ts_freq):
    """Check input frequency (main dataframe) with ts_freq.

    check 1- Determine whether input ts_freq is valid or not.
    check 2- Compare the frequency of main dataframe with 'ts_freq'.

    Parameters
    ----------
    x_df : DataFrame
        Dataframe with time index
    datetime_freq : str
        Frequency of main dataframe (x_df)
    ts_freq : str
        selected frequency by user, can be one of the following:
        {"H", "D", "W", "M", "Q", "Y"}

    Raises
    ------
    ValueError
        Raise an error if:
        1- input ts_freq is invalid and not in the specified dict i.e.,
        {"H", "D", "W", "M", "Q", "Y"}
        2- Level of base dataframe ts_frequency  is lower than selected ts_freq
    """
    # Check 1: Determine whether input ts_freq is valid or not
    freq_list = datetime_freq["frequency"].tolist()
    if (ts_freq is not None) & (ts_freq not in freq_list):
        raise ValueError(f"Invalid ts_freq specified, must be in: {freq_list}")

    # Check 2: Compare the frequency of main dataframe with 'ts_freq'
    # 2-1: Determine frequency of main DataFrame, find in ranking
    main_df_datetime_freq = pd.infer_freq(x_df["date_sequence"])[0]
    rank_main_df = datetime_freq.loc[
        datetime_freq["frequency"] == main_df_datetime_freq, "rank"
    ].max()
    rank_ts_freq = datetime_freq.loc[
        datetime_freq["frequency"] == ts_freq, "rank"
    ].max()
    # 2-2: Compare the frequency of main df with 'ts_freq'
    if rank_main_df < rank_ts_freq:
        raise ValueError(
            "Level of base dataframe ts_frequency  is lower than selected ts_freq"
        )
