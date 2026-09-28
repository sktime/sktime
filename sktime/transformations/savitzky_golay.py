# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Savitzky-Golay filter transformer for time series."""

__author__ = ["ved197338"]
__all__ = ["SavitzkyGolayTransformer"]

import pandas as pd
from scipy.signal import savgol_filter

from sktime.transformations.base import BaseTransformer


class SavitzkyGolayTransformer(BaseTransformer):
    """Savitzky-Golay filter for smoothing or differentiating time series.

    Uses local polynomial regression (convolution) to smooth data or to
    compute numerical derivatives, preserving features of the distribution
    like relative maxima and minima better than moving average approaches.

    Wraps ``scipy.signal.savgol_filter``.

    Parameters
    ----------
    window_length : int, default=5
        Length of the filter window. Must be a positive odd integer.
    polyorder : int, default=2
        Order of the polynomial used to fit the samples.
        Must be less than ``window_length``.
    deriv : int, default=0
        Order of the derivative to compute.
        Use 0 to simply smooth the data without differentiation.
    delta : float, default=1.0
        Spacing of the samples to which the filter will be applied.
        Only relevant when ``deriv > 0``.
    mode : str, default="interp"
        How to extend the signal at the boundaries.
        One of ``"interp"``, ``"mirror"``, ``"nearest"``,
        ``"wrap"``, ``"constant"``.
    cval : float, default=0.0
        Value to fill past the edges of the input when
        ``mode`` is ``"constant"``.

    See Also
    --------
    scipy.signal.savgol_filter : The underlying scipy implementation.

    Examples
    --------
    >>> from sktime.transformations.savitzky_golay import (
    ...     SavitzkyGolayTransformer,
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> t = SavitzkyGolayTransformer(window_length=7, polyorder=2)
    >>> y_smooth = t.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ved197338"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "capability:multivariate": True,
        "capability:categorical_in_X": False,
    }

    def __init__(
        self,
        window_length=5,
        polyorder=2,
        deriv=0,
        delta=1.0,
        mode="interp",
        cval=0.0,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.mode = mode
        self.cval = cval
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed.
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            Smoothed or differentiated version of X.
        """
        self._check_params()

        # work out an effective window length that fits the data
        w = self._effective_window_length(len(X))
        p = min(self.polyorder, w - 1)

        if isinstance(X, pd.Series):
            out = savgol_filter(
                X.to_numpy(),
                window_length=w,
                polyorder=p,
                deriv=self.deriv,
                delta=self.delta,
                mode=self.mode,
                cval=self.cval,
                axis=0,
            )
            return pd.Series(out, index=X.index, name=X.name)

        # pd.DataFrame branch
        out = savgol_filter(
            X.to_numpy(),
            window_length=w,
            polyorder=p,
            deriv=self.deriv,
            delta=self.delta,
            mode=self.mode,
            cval=self.cval,
            axis=0,
        )
        return pd.DataFrame(out, index=X.index, columns=X.columns)

    def _effective_window_length(self, n_samples):
        """Adjust window length so it fits the actual data length."""
        if n_samples <= self.polyorder:
            raise ValueError(
                f"Data length ({n_samples}) must be greater than "
                f"`polyorder` ({self.polyorder})."
            )

        w = min(self.window_length, n_samples)
        # savgol_filter requires an odd window length
        if w % 2 == 0:
            w -= 1
        # window must be larger than polyorder
        if w <= self.polyorder:
            w = self.polyorder + 1
            if w % 2 == 0:
                w += 1
        if w > n_samples:
            raise ValueError(
                f"Effective window length ({w}) exceeds data length ({n_samples}). "
                f"Data length must be > polyorder ({self.polyorder})."
            )
        return w

    def _check_params(self):
        """Validate user-supplied parameters."""
        if not isinstance(self.window_length, int) or self.window_length <= 0:
            raise ValueError(
                f"`window_length` must be a positive integer, got {self.window_length}"
            )
        if self.window_length % 2 == 0:
            raise ValueError(f"`window_length` must be odd, got {self.window_length}")
        if not isinstance(self.polyorder, int) or self.polyorder < 0:
            raise ValueError(
                f"`polyorder` must be a non-negative integer, got {self.polyorder}"
            )
        if self.polyorder >= self.window_length:
            raise ValueError(
                f"`polyorder` ({self.polyorder}) must be less than "
                f"`window_length` ({self.window_length})"
            )
        if not isinstance(self.deriv, int) or self.deriv < 0:
            raise ValueError(
                f"`deriv` must be a non-negative integer, got {self.deriv}"
            )
        if not isinstance(self.delta, (int, float)) or self.delta <= 0:
            raise ValueError(f"`delta` must be a positive number, got {self.delta}")
        valid_modes = {"interp", "mirror", "nearest", "wrap", "constant"}
        if self.mode not in valid_modes:
            raise ValueError(f"`mode` must be one of {valid_modes}, got '{self.mode}'")
        if not isinstance(self.cval, (int, float)):
            raise ValueError(f"`cval` must be a number, got {type(self.cval).__name__}")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.
            If no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {"window_length": 5, "polyorder": 2, "deriv": 0}
        params2 = {"window_length": 7, "polyorder": 3, "deriv": 1, "delta": 0.5}
        return [params1, params2]
