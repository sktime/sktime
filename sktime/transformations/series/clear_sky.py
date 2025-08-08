# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Clear sky transformer for solar time-series."""

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

# todo: update function?
# todo: clock changes, time-zone aware index, milliseconds?


class ClearSky(BaseTransformer):
    """Clear sky transformer for solar data.

    This is a transformation which converts a time series from it's original
    domain into a percentage domain. The numerator at each time step in the
    transformation is the input values, the denominator is a weighted
    quantile of the time series for that particular time step. In the example
    of solar power transformations, the denominator is an approximation of the
    clear sky power, and the output of the transformation is the clearness index.

    The clear sky power, i.e. the denominator, is calculated on a grid containing
    each unique combination of time-of-day and day-of-year. The spacing of the
    grid depends on the frequency of the input data.

    The weights are defined using von-mises kernels with bandwidths chosen by the
    user.

    This transformation can be inaccurate at low values, in the solar example during
    early morning and late evening. Therefore, clear sky values below a threshold can
    be fixed to zero in the transformed domain. Denominator values of zero are set
    to zero in the transformed domain by default.

    This transformer is based on the work detailed in [1]_.

    Parameters
    ----------
    quantile_prob : float, default=0.95
        The probability level used to calculate the weighted quantile
    bw_diurnal : float, default=100
        The bandwidth of the diurnal kernel. This is the kappa value of the
        von mises kernel for time of day.
    bw_annual : float, default=10
        The bandwidth of the annual kernel. This is the kappa value of the
        von mises kernel for day of year.
    min_thresh : float, default=0
        The threshold of the clear sky power below which values are
        set to zero in the transformed domain.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    backend : str, default="loky"
        Specify the parallelisation backend implementation in joblib, where
        "loky" is used by default.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.solener.2009.05.016

    Examples
    --------
    >>> from sktime.transformations.series.clear_sky import ClearSky  # doctest: +SKIP
    >>> from sktime.datasets import load_solar  # doctest: +SKIP
    >>> y = load_solar()  # doctest: +SKIP
    >>> transformer = ClearSky()  # doctest: +SKIP
    >>> # takes ~1min
    >>> y_trafo = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ciaran-g"],
        "maintainers": ["ciaran-g"],
        "python_dependencies": ["statsmodels", "joblib", "scipy"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "X_inner_mtype": [
            "pd.Series",
        ],  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": [
            pd.DatetimeIndex,
            pd.PeriodIndex,
        ],  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        "capability:unequal_length:removes": True,  # ?
        "capability:missing_values": False,
        "capability:missing_values:removes": True,
    }

    def __init__(
        self,
        quantile_prob=0.95,
        bw_diurnal=100,
        bw_annual=10,
        min_thresh=0,
        n_jobs=None,
        backend="loky",
    ):
        self.quantile_prob = quantile_prob
        self.bw_diurnal = bw_diurnal
        self.bw_annual = bw_annual
        self.min_thresh = min_thresh
        self.n_jobs = n_jobs
        self.backend = backend

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series of pd.DataFrame
            Data used to estimate the clear sky power.
        y : Ignored argument for interface compatibility.

        Returns
        -------
        self: reference to self
        """
        from joblib import Parallel, delayed

        # check that the data is formatted correctly etc
        self.freq = _check_index(X)
        # now get grid of model
        df = pd.DataFrame(index=X.index)
        df["yday"] = df.index.dayofyear
        df["tod"] = df.index.hour + df.index.minute / 60 + df.index.second / 60

        # set up smoothing grid
        tod = pd.timedelta_range(start="0T", end="1D", freq=self.freq)[:-1]
        tod = [(x.total_seconds() / (60 * 60)) for x in tod.to_pytimedelta()]
        yday = pd.RangeIndex(start=1, stop=367)
        indx = pd.MultiIndex.from_product([yday, tod], names=["yday", "tod"])

        # set up parallel function and backend
        parallel = Parallel(n_jobs=self.n_jobs, backend=self.backend)

        def par_csp(x):
            res = _clearskypower(
                y=X,
                q=self.quantile_prob,
                tod_i=x[1],
                doy_i=x[0],
                tod_vec=df["tod"],
                doy_vec=df["yday"],
                bw_tod=self.bw_diurnal,
                bw_doy=self.bw_annual,
            )

            return res

        # calculate the csp
        csp = parallel(delayed(par_csp)(name) for name in indx)
        csp = pd.Series(csp, index=indx, dtype="float64")
        self.clearskypower = csp.sort_index()

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series of pd.DataFrame
            Data used to be transformed.
        y : Ignored argument for interface compatibility.

        Returns
        -------
        X_trafo : transformed version of X
        """
        _freq_ind = _check_index(X)
        if self.freq != _freq_ind:
            raise ValueError(
                """
                Change in frequency detected from original input. Make sure
                X is the same frequency as used in .fit().
                """
            )
        # get required seasonal index
        yday = X.index.dayofyear
        tod = X.index.hour + X.index.minute / 60 + X.index.second / 60
        indx_seasonal = pd.MultiIndex.from_arrays([yday, tod], names=["yday", "tod"])

        # look up values and overwrite index
        csp = self.clearskypower[indx_seasonal].copy()
        csp.index = X.index
        X_trafo = X / csp

        # threshold for small morning/evening values
        X_trafo[(csp <= self.min_thresh) & (X.notnull())] = 0

        return X_trafo

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from
        inverse_transform

        Parameters
        ----------
        X : Series of pd.DataFrame
            Data used to be inversed transformed.
        y : Ignored argument for interface compatibility.

        Returns
        -------
        X_trafo : inverse transformed version of X
        """
        _freq_ind = _check_index(X)
        if self.freq != _freq_ind:
            raise ValueError(
                """
                Change in frequency detected from original input. Make sure
                X is the same frequency as used in .fit().
                """
            )
        yday = X.index.dayofyear
        tod = X.index.hour + X.index.minute / 60 + X.index.second / 60
        indx_seasonal = pd.MultiIndex.from_arrays([yday, tod], names=["yday", "tod"])

        # look up values and overwrite index
        csp = self.clearskypower[indx_seasonal].copy()
        csp.index = X.index
        X_trafo = X * csp

        return X_trafo

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        params = {"clearskypower": self.clearskypower}

        return params

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
        params = {
            "quantile_prob": 0.95,
            "bw_diurnal": 100,
            "bw_annual": 10,
            "min_thresh": None,
        }

        return params


def _clearskypower(y, q, tod_i, doy_i, tod_vec, doy_vec, bw_tod, bw_doy):
    """Estimate the clear sky power for a given day-of-year and hour-of-day.

    Parameters
    ----------
    y : Series of measurements
    q : Probability level used for the quantile
    tod_i : time-of-day of interest in hours
    doy_i : day-of-year of interest in days
    tod_vec : Series of time-of-day corresponding to the index of y
    doy_vec: Series of day-of-year corresponding to the index of y
    bw_tod : Kappa value used for defining weights for time-of-day
    bw_doy : Kappa value used for defining weights for day-of-year

    Returns
    -------
    csp : float
        The clear sky power at tod_i and doy_i
    """
    from scipy.stats import vonmises
    from statsmodels.stats.weightstats import DescrStatsW

    wts_tod = vonmises.pdf(
        x=tod_i * 2 * np.pi / 24, kappa=bw_tod, loc=tod_vec * 2 * np.pi / 24
    )
    wts_doy = vonmises.pdf(
        x=doy_i * 2 * np.pi / 365.25, kappa=bw_doy, loc=doy_vec * 2 * np.pi / 365.25
    )

    wts = wts_doy * wts_tod
    wts = wts / wts.sum()

    csp = DescrStatsW(y, weights=wts).quantile(probs=q).values[0]

    return csp


def _check_index(X):
    """Check input value frequency is set and we have the correct index.

    Parameters
    ----------
    X : Series or pd.DataFrame
        Data used to be inversed transformed.

    Raises
    ------
    ValueError : Input index must be class pd.DatetimeIndex or pd.PeriodIndex.
    ValueError : Input index frequency cannot be inferred and is not set.
    ValueError : Frequency of data not suitable for transformer as is.

    Returns
    -------
    freq_ind : str or None
        Frequency of data in string format
    """
    if not (isinstance(X.index, pd.DatetimeIndex)) | (
        isinstance(X.index, pd.PeriodIndex)
    ):
        raise ValueError(
            "Input index must be class pd.DatetimeIndex or pd.PeriodIndex."
        )
    # check that it has a frequency, if not infer
    freq_ind = X.index.freq
    if freq_ind is None:
        freq_ind = pd.infer_freq(X.index)
        if freq_ind is None:
            raise ValueError("Input index frequency cannot be inferred and is not set.")

    tod = pd.timedelta_range(start="0T", end="1D", freq=freq_ind)
    # check frequency of tod
    if (tod.freq > pd.offsets.Day(1)) | (tod.freq < pd.offsets.Second(1)):
        raise ValueError(
            """
            Transformer intended to be used with input frequency of greater than
            or equal to one day and with a frequency of less or equal to than
            1 second. Contributions welcome on adapting for these use cases.
            """
        )
    return freq_ind
