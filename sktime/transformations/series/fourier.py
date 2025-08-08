# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Fourier features for time series with long/complex seasonality."""

__author__ = ["ltsaprounis", "blazingbhavneek"]

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.fft import rfft

from sktime.transformations.base import BaseTransformer


class FourierFeatures(BaseTransformer):
    r"""Fourier Features for time series seasonality.

    Fourier Series terms can be used as explanatory variables for the cases of multiple
    seasonal periods and or complex / long seasonal periods [1]_, [2]_. For every
    seasonal period, :math:`sp` and fourier term :math:`k` pair there are 2 fourier
    terms sin_sp_k and cos_sp_k:
        - sin_sp_k = :math:`sin(\frac{2 \pi k t}{sp})`
        - cos_sp_k = :math:`cos(\frac{2 \pi k t}{sp})`

    Where :math:`t` is the elapsed time since the beginning of the seasonal period and
    :math:`sp` the total time of the seasonal period.

    The transformed output is a series that contains all requested Fourier terms.

    Warning: the output will contain only the Fourier terms under default settings,
    and discard the original columns of the input data, to avoid multiplication
    of the original data in a pipeline or ``FeatureUnion``.
    To keep the original columns, set ``keep_original_columns=True``.

    Names of the columns are generated as follows:
    additional columns with the naming convention stated above (sin_sp_k and cos_sp_k).
    The numbers of Fourier terms :math:`K` in the fourier_terms_list
    determines the number of Fourier terms that will be used for each seasonal period,
    i.e., Fourier terms :math:`k = 1\dots K` (integers), cos and sine, will be generated
    for the seasonality :math:`sp` at the same list index.
    For example, consider sp_list = [12, "Y"] and fourier_terms_list = [2, 1].
    This says that we compute 2 (2 cos, 2 sine) Fourier terms for
    seasonality 12 periods, and 1 Fourier term (1 cos and 1 sine)
    for seasonality 1 year.
    The transformed series will then have columns with the following names:
    "cos_12_1", "sin_12_1", "cos_12_2", "sin_12_2", "cos_Y_1", "sin_Y_1"

    The implementation is based on the fourier function from the R forecast package [3]_

    Parameters
    ----------
    sp_list : List[float and/or str]
        List of seasonal periods. Can be defined with the following options:

        * | float : Periodicity defined as number of timesteps since the beginning of
            the data seen in ``fit``.

        * | string : Periodicity defined as a column name in X that contains the
            :math:`t/sp` values.

        * | string : Periodicity defined as a pandas period alias:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases

    fourier_terms_list : List[int]
        List of number of fourier terms (:math:`K`) per corresponding (:math:`sp`); each
        :math:`K` matches to one :math:`sp` of the sp_list. For example, if sp_list =
        [7, "Y"] and fourier_terms_list = [3, 9], the seasonality of 7 timesteps will
        have 3 sin_sp_k and 3 cos_sp_k fourier terms and the yearly seasonality "Y" will
        have 9 sin_sp_k and 9 cos_sp_k fourier terms.
    freq : str, optional, default = None
        Only used when X has a pd.DatetimeIndex without a specified frequency.
        Specifies the frequency of the index of your data. The string should
        match a pandas offset alias:

        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    keep_original_columns : boolean, optional, default=False
        Keep original columns in X passed to ``.transform()``

    References
    ----------
    .. [1] Hyndsight - Forecasting with long seasonal periods:
        https://robjhyndman.com/hyndsight/longseasonality/
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3.
        Accessed on August 14th 2022.
    .. [3] https://pkg.robjhyndman.com/forecast/reference/fourier.html

    Examples
    --------
    >>> from sktime.transformations.series.fourier import FourierFeatures
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = FourierFeatures(sp_list=[12, "Y"], fourier_terms_list=[4, 1])
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ltsaprounis", "blazingbhavneek"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": [
            pd.PeriodIndex,
            pd.DatetimeIndex,
        ],  # index type that needs to be enforced
        # in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        #   not relevant for transformers that return Primitives in transform-output
        "capability:missing_values": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
    }

    def __init__(
        self,
        sp_list: list[float],
        fourier_terms_list: list[int],
        freq: Optional[str] = None,
        keep_original_columns: Optional[bool] = False,
    ):
        self.sp_list = sp_list
        self.fourier_terms_list = fourier_terms_list
        self.freq = freq
        self.keep_original_columns = keep_original_columns

        if len(self.sp_list) != len(self.fourier_terms_list):
            raise ValueError(
                "In FourierFeatures the length of the sp_list needs to be equal "
                "to the length of fourier_terms_list."
            )

        for i in range(len(self.sp_list)):
            if (
                not isinstance(sp_list[i], str)
                and sp_list[i] / fourier_terms_list[i] < 1
            ):
                raise ValueError(
                    "In FourierFeatures the number of each element of "
                    "fourier_terms_list needs to be lower from the corresponding "
                    "element of the sp_list"
                )

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        # Create the sp, k pairs
        # Don't add pairs where the coefficient k/sp already exists
        self.sp_k_pairs_list_ = []
        coefficient_list = []
        for i, sp in enumerate(self.sp_list):
            for k in range(1, self.fourier_terms_list[i] + 1):
                if not isinstance(sp, str):  # periodicity sp relative to start
                    coef = k / sp
                    if coef not in coefficient_list:
                        coefficient_list.append(coef)
                        self.sp_k_pairs_list_.append((sp, k))
                    else:
                        warnings.warn(
                            f"The terms sin_{sp}_{k} and cos_{sp}_{k} from "
                            "FourierFeatures will be skipped because the resulting "
                            "coefficient already exists from other seasonal period, "
                            "fourier term pairs.",
                            stacklevel=2,
                        )
                else:  # periodicity sp from offset string or X column
                    self.sp_k_pairs_list_.append((sp, k))

        time_index = X.index

        if isinstance(time_index, pd.DatetimeIndex):
            # Chooses first non None value
            self.freq_ = time_index.freq or self.freq or pd.infer_freq(time_index)
            if self.freq_ is None:
                raise ValueError("X has no known frequency and none is supplied")
            if self.freq_ == time_index.freq and self.freq_ != self.freq:
                warnings.warn(
                    f"Using frequency from index: {time_index.freq}, which "
                    f"does not match the frequency given:{self.freq}.",
                    stacklevel=2,
                )
            time_index = time_index.to_period(self.freq_)
        # this is used to make sure that time t is calculated with reference to
        # the data passed on fit
        # store the integer form of the minimum date in the prediod index
        self.min_t_ = np.min(time_index.astype("int64"))

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        X_transformed = pd.DataFrame(index=X.index)
        X_df = pd.DataFrame(X)

        if isinstance(X.index, pd.DatetimeIndex):
            time_index = X.index.to_period(self.freq_)
        else:
            time_index = X.index

        # get the integer form of the PeriodIndex
        int_index = time_index.astype("int64") - self.min_t_

        for sp_k in self.sp_k_pairs_list_:
            sp = sp_k[0]
            k = sp_k[1]

            if not isinstance(sp, str):  # periodicity sp relative to start
                X_transformed[f"sin_{sp}_{k}"] = np.sin(int_index * 2 * k * np.pi / sp)
                X_transformed[f"cos_{sp}_{k}"] = np.cos(int_index * 2 * k * np.pi / sp)

            elif sp in X_df.columns:  # periodicity sp from X column
                frac_index = X_df[sp].values
                X_transformed[f"sin_{sp}_{k}"] = np.sin(frac_index * 2 * k * np.pi)
                X_transformed[f"cos_{sp}_{k}"] = np.cos(frac_index * 2 * k * np.pi)

            else:  # periodicity sp from offset string
                if isinstance(X.index, pd.PeriodIndex):
                    datetime_index = X.index.to_timestamp()
                else:
                    datetime_index = X.index

                frac_index = self._offset_frac_since_prev_offset(
                    datetime_index=datetime_index,
                    period_str=sp,
                )
                X_transformed[f"sin_{sp}_{k}"] = np.sin(frac_index * 2 * k * np.pi)
                X_transformed[f"cos_{sp}_{k}"] = np.cos(frac_index * 2 * k * np.pi)

        if self.keep_original_columns:
            X_transformed = pd.concat([X, X_transformed], axis=1, copy=True)

        return X_transformed

    def _offset_frac_since_prev_offset(self, datetime_index, period_str):
        """Get time passed as fraction of the current period.

        Parameters
        ----------
        datetime_index : pandas DatetimeIndex
        period_str : pandas period str
            Cannot contain digits

        Returns
        -------
        numpy array containing the time passed between [previous offset, next offset)
        as fraction in the interval [0, 1) for every datetime in datetimes
        """

        def _get_frac(datetime, offset_boundaries):
            i = np.searchsorted(offset_boundaries, datetime, side="right")
            prev = offset_boundaries[i - 1]
            next = offset_boundaries[i]
            period_timedelta = next - prev
            since_prev_timedelta = datetime - prev
            return since_prev_timedelta / period_timedelta

        offset = pd.tseries.frequencies.to_offset(period_str)
        offset_boundaries = pd.date_range(
            start=np.amin(datetime_index) - offset,
            end=np.amax(datetime_index) + offset,
            freq=period_str,
            tz=datetime_index.tz,
        )

        # date_range created with offsets <= 1day have boundaries on the first
        # moment of the new period, but date_range created with offsets > 1day
        # have boundaries on the last day of the period rather than the desired
        # first day of new period. workaround: shift by 1 day
        offset_td = pd.to_timedelta(offset, errors="coerce")
        if not offset_td <= pd.Timedelta(days=1):
            offset_boundaries = offset_boundaries + pd.Timedelta(days=1)

        fracs = [_get_frac(dt, offset_boundaries) for dt in datetime_index]

        return np.array(fracs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"sp_list": [12], "fourier_terms_list": [4]},
            {"sp_list": [12, 6.2], "fourier_terms_list": [3, 4]},
            {"sp_list": ["Y"], "fourier_terms_list": [4]},
            {"sp_list": ["Y", "Q"], "fourier_terms_list": [3, 4]},
        ]
        return params


class FourierTransform(BaseTransformer):
    r"""Simple Fourier transform for time series.

    The implementation is based on the real fast fourier transform from numpy.fft.rfft
    Returns pd.Series of amplitudes of integer range frequencies.
    Even-Sampling of data is assumed and frequency range converted to integer.

    Examples
    --------
    >>> from sktime.transformations.series.fourier import FourierTransform
    >>> from sktime.datasets import load_airline
    >>> X = load_airline()
    >>> transformer = FourierTransform()
    >>> X_ft = transformer.fit_transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "univariate-only": True,
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series mtype X_inner_mtype

        Returns
        -------
        transformed version of X
        """
        # numpy.fft methods
        dft_seq = np.abs(rfft(X))

        # Combining the arrays to Pandas Series
        Y = pd.Series(dft_seq[1:])
        return Y
