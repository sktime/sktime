# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for seasonality - periodogram."""

__all__ = ["SeasonalityPeriodogram"]

import numpy as np

from sktime.param_est.base import BaseParamFitter


class SeasonalityPeriodogram(BaseParamFitter):
    """Score periodicities by their spectral power.

    Computes seasonality periodogram based on ``iloc`` indices (not ``loc`` labels),
    and finds significant periods based on their spectral power, using
    Welch's method of periodogram averaging [1]_.

    Computes significant periods based on a threshold of the maximum power,
    i.e., periods with power above ``thresh * maxpower`` are considered significant,
    and the one with highest power is considered the main seasonality period.

    Significance is determined by thresholding as above, not by statistical testing.

    Based on ``seasonal`` package by ``welch`` [2]_.

    Parameters
    ----------
    min_period : int
        Disregard periods shorter than this number of samples.
        Defaults to 4
    max_period : int
        Disregard periods longer than this number of samples.
        Defaults to None
    thresh : float (0..1)
        Retain periods scoring above thresh*maxscore. Defaults to 0.10

    Attributes
    ----------
    sp_ : int, seasonality period with highest power, if any sub-threshold, else 1
    sp_significant_ : list of int, array of Fourier periods in descending order
        of their powers.

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Welch%27s_method
    .. [2]: https://github.com/welch/seasonal

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.seasonality import SeasonalityPeriodogram
    >>> X = load_airline().diff()[1:]
    >>> sp_est = SeasonalityPeriodogram()
    >>> sp_est.fit(X)
    SeasonalityPeriodogram(...)
    >>> sp_est.get_fitted_params()["sp"]
    6
    >>> sp_est.get_fitted_params()["sp_significant"]
    array([ 6, 12, 14,  4, 10,  5])
    """

    _tags = {
        "authors": ["welch", "blazingbhavneek", "JATAYU000"],
        "maintainers": ["blazingbhavneek"],
        "X_inner_mtype": "pd.Series",
        "scitype:X": "Series",
        "capability:missing_values": True,
        "capability:multivariate": False,
    }

    def __init__(self, min_period=4, max_period=None, thresh=0.10):
        self.min_period = min_period
        self.max_period = max_period
        self.thresh = thresh
        self.MIN_FFT_CYCLES = 3.0
        self.MAX_FFT_PERIOD = 512
        super().__init__()

    def periodogram_peaks(self, data, min_period=4, max_period=None, thresh=0.90):
        """Return a list of intervals containing high-scoring periods.

        Use a robust periodogram to estimate ranges containing
        high-scoring periodicities in possibly short, noisy data. Returns
        each peak period along with its adjacent bracketing periods from
        the FFT coefficient sequence.

        Data should be detrended for sharpest results, but trended data
        can be accommodated by lowering thresh (resulting in more
        intervals being returned)

        Code adapted from the standalone ``seasonal`` package by ``welch``.

        Parameters
        ----------
        data : 1D ndarray
            Data series, evenly spaced samples.
        min_period : int, optional, default=4
            Disregard periods shorter than this number of samples.
        max_period : int, optional, default = None = as below
            Disregard periods longer than this number of samples.
            Defaults to the smaller of len(data)/MIN_FFT_CYCLES or MAX_FFT_PERIOD
        thresh : float (0..1), optional, default=0.9
            Retain periods scoring above thresh*maxscore.

        Returns
        -------
        periods : array of quads, or None
            Array of (period, power, period-, period+), maximizing period
            and its score, and FFT periods bracketing the maximizing
            period, returned in decreasing order of score

        Notes
        -----
        This method is adapted from the standalone seasonal package.

        References
        ----------
        .. [1] https://github.com/welch/seasonal
        """
        periods, power = self.periodogram(data, min_period, max_period)
        if np.all(np.isclose(power, 0.0)):
            return None  # DC
        result = []
        keep = power.max() * thresh
        while True:
            peak_i = power.argmax()
            if power[peak_i] < keep:
                break
            min_period = periods[min(peak_i + 1, len(periods) - 1)]
            max_period = periods[max(peak_i - 1, 0)]
            result.append([periods[peak_i], power[peak_i], min_period, max_period])
            power[peak_i] = 0
        return result if len(result) else None

    def periodogram(self, data, min_period=4, max_period=None):
        """Score periodicities by their spectral power.

        Produce a robust periodogram estimate for each possible periodicity
        of the (possibly noisy) data.

        Code adapted from the standalone ``seasonal`` package by ``welch``.

        Parameters
        ----------
        data : 1D ndarray
            Data series, having at least three periods of data.
        min_period : int, optional, default=4
            Disregard periods shorter than this number of samples.
        max_period : int, optional, default = None = as below
            Disregard periods longer than this number of samples.
            Defaults to the smaller of len(data)/MIN_FFT_CYCLES or MAX_FFT_PERIOD

        Returns
        -------
        periods, power : ndarray, ndarray
            Periods is an array of Fourier periods in descending order,
            beginning with the first one greater than max_period.
            Power is an array of spectral power values for the periods

        Notes
        -----
        This uses Welch's method (no relation) of periodogram
        averaging[1]_, which trades off frequency precision for better
        noise resistance. We don't look for sharp period estimates from
        it, as it uses the FFT, which evaluates at periods N, N/2, N/3, ...,
        so that longer periods are sparsely sampled.

        References
        ----------
        .. [1]: https://en.wikipedia.org/wiki/Welch%27s_method
        .. [2]: https://github.com/welch/seasonal
        """
        import scipy.signal

        data = np.asarray(data)
        if max_period is None:
            max_period = int(min(len(data) / self.MIN_FFT_CYCLES, self.MAX_FFT_PERIOD))
        nperseg = min(max_period * 2, len(data) // 2)  # FFT window
        freqs, power = scipy.signal.welch(
            data, 1.0, scaling="spectrum", nperseg=nperseg
        )
        periods = np.array([int(round(1.0 / freq)) for freq in freqs[1:]])
        power = power[1:]
        # take the max among frequencies having the same integer part
        idx = 1
        while idx < len(periods):
            if periods[idx] == periods[idx - 1]:
                power[idx - 1] = max(power[idx - 1], power[idx])
                periods, power = np.delete(periods, idx), np.delete(power, idx)
            else:
                idx += 1
        power[periods == nperseg] = 0  # disregard the artifact at nperseg
        min_i = len(periods[periods >= max_period]) - 1
        max_i = len(periods[periods < min_period])
        periods, power = periods[min_i:-max_i], power[min_i:-max_i]
        return periods, power

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        seasons = self.periodogram_peaks(
            X,
            min_period=self.min_period,
            max_period=self.max_period,
            thresh=self.thresh,
        )

        if seasons is None or len(seasons) == 0:
            self.sp_ = 1
            self.sp_significant_ = []
        else:
            seasons = [int(x[0]) for x in seasons]
            self.sp_significant_ = np.array(seasons)
            self.sp_ = int(self.sp_significant_[0])

        return self

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
        params1 = {}
        params2 = {"min_period": 5, "max_period": 24, "thresh": 0.1}
        params3 = {"min_period": 5}

        return [params1, params2, params3]
