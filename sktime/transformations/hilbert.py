# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hilbert transform feature extractor for time series."""

__author__ = ["ved197338"]
__all__ = ["HilbertTransformer"]

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from sktime.transformations.base import BaseTransformer


class HilbertTransformer(BaseTransformer):
    """Extract instantaneous features from a time series via Hilbert transform.

    Computes the analytic signal using the Hilbert transform and returns
    one of several derived representations: the amplitude envelope,
    instantaneous phase, instantaneous frequency, the quadrature
    component, or all three main features as a multi-column DataFrame.

    Wraps ``scipy.signal.hilbert``.

    Parameters
    ----------
    output_type : str, default="envelope"
        Which feature to extract. One of:

        - ``"envelope"`` : instantaneous amplitude, ``|z(t)|``
        - ``"phase"`` : unwrapped instantaneous phase in radians
        - ``"frequency"`` : instantaneous frequency in cycles per sample
        - ``"quadrature"`` : imaginary part of the analytic signal
        - ``"all"`` : envelope, phase and frequency as separate columns

    N : int or None, default=None
        Number of Fourier components (FFT length).
        If ``None``, defaults to the length of the input.
    unwrap_phase : bool, default=True
        Whether to unwrap the phase to remove 2-pi discontinuities.
    fs : float, default=1.0
        Sampling frequency, used to scale instantaneous frequency
        into physical units (Hz) when ``output_type`` is
        ``"frequency"`` or ``"all"``.

    See Also
    --------
    scipy.signal.hilbert : The underlying scipy implementation.

    Examples
    --------
    >>> from sktime.transformations.hilbert import HilbertTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> t = HilbertTransformer(output_type="envelope")
    >>> y_envelope = t.fit_transform(y)
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
        output_type="envelope",
        N=None,
        unwrap_phase=True,
        fs=1.0,
    ):
        self.output_type = output_type
        self.N = N
        self.unwrap_phase = unwrap_phase
        self.fs = fs
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
        Xt : pd.Series or pd.DataFrame
            Hilbert feature(s) extracted from X.
            When ``output_type="all"`` the output is always a DataFrame
            with ``_envelope``, ``_phase``, ``_frequency`` suffixed columns.
        """
        self._check_params()

        was_series = isinstance(X, pd.Series)
        df = pd.DataFrame(X) if was_series else X
        n = len(df)

        if self.N is not None and self.N < n:
            raise ValueError(
                f"`N` ({self.N}) must be greater than or equal to signal length ({n})."
            )

        result = {}

        for col in df.columns:
            arr = df[col].to_numpy()

            # compute the analytic signal; trim back to original length
            # in case N was specified larger than n
            z = hilbert(arr, N=self.N, axis=0)[:n]

            envelope = np.abs(z)
            phase = np.angle(z)
            if self.unwrap_phase:
                phase = np.unwrap(phase)

            if self.output_type == "envelope":
                result[col] = envelope
            elif self.output_type == "quadrature":
                result[col] = np.imag(z)
            elif self.output_type == "phase":
                result[col] = phase
            elif self.output_type == "frequency":
                result[col] = np.gradient(phase) * self.fs / (2 * np.pi)
            elif self.output_type == "all":
                freq = np.gradient(phase) * self.fs / (2 * np.pi)
                result[f"{col}_envelope"] = envelope
                result[f"{col}_phase"] = phase
                result[f"{col}_frequency"] = freq

        out = pd.DataFrame(result, index=df.index)

        # if the input was a Series and we didn't fan out into multiple
        # columns, return a Series to keep things consistent
        if was_series and self.output_type != "all":
            return pd.Series(out.iloc[:, 0], index=X.index, name=X.name)

        return out

    def _check_params(self):
        """Validate user-supplied parameters."""
        valid = {"envelope", "phase", "frequency", "quadrature", "all"}
        if self.output_type not in valid:
            raise ValueError(
                f"`output_type` must be one of {valid}, got '{self.output_type}'"
            )
        if self.N is not None:
            if not isinstance(self.N, int) or self.N <= 0:
                raise ValueError(
                    f"`N` must be a positive integer or None, got {self.N}"
                )
        if not isinstance(self.unwrap_phase, bool):
            raise ValueError(
                f"`unwrap_phase` must be True or False, got {self.unwrap_phase!r}"
            )
        if not isinstance(self.fs, (int, float)) or self.fs <= 0:
            raise ValueError(f"`fs` must be a positive number, got {self.fs}")

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
        params1 = {"output_type": "envelope"}
        params2 = {"output_type": "all", "unwrap_phase": True, "fs": 2.0}
        params3 = {"output_type": "frequency"}
        return [params1, params2, params3]
