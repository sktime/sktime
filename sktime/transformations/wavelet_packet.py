# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Wavelet Packet Decomposition feature transformer."""

__author__ = ["ved197338"]
__all__ = ["WaveletPacketTransformer"]

import math

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class WaveletPacketTransformer(BaseTransformer):
    """Wavelet Packet Decomposition transformer.

    Unlike the standard DWT which only decomposes the approximation branch
    at each level, wavelet packet decomposition recursively decomposes
    *both* approximation and detail branches, giving ``2**level`` terminal
    sub-bands. This provides a richer frequency resolution.

    Currently uses Haar wavelet coefficients internally.

    Parameters
    ----------
    level : int, default=2
        Number of decomposition levels.
        Produces ``2**level`` terminal sub-band nodes.
    output_feature : str, default="energy"
        What to extract from each sub-band. One of:

        - ``"energy"`` : sum of squared coefficients per node
        - ``"entropy"`` : Shannon entropy of normalized coefficient power
        - ``"coefficients"`` : concatenated raw packet coefficients

    See Also
    --------
    sktime.transformations.dwt.DWTTransformer :
        Standard Discrete Wavelet Transform (decomposes
        approximation branch only).

    Examples
    --------
    >>> from sktime.transformations.wavelet_packet import (
    ...     WaveletPacketTransformer,
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> t = WaveletPacketTransformer(level=2, output_feature="energy")
    >>> y_features = t.fit_transform(y)
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
        "transform-returns-same-time-index": False,
        "capability:multivariate": True,
        "capability:categorical_in_X": False,
    }

    def __init__(self, level=2, output_feature="energy"):
        self.level = level
        self.output_feature = output_feature
        super().__init__()

    def __dynamic_tags__(self):
        """Set dynamic output scitype based on output_feature."""
        if self.output_feature in ("energy", "entropy"):
            self.set_tags(**{"scitype:transform-output": "Primitives"})
        else:
            self.set_tags(**{"scitype:transform-output": "Series"})

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
            Extracted features. For ``"energy"`` and ``"entropy"`` this is
            a single-row DataFrame with one column per sub-band node.
            For ``"coefficients"`` the output preserves the original length.
        """
        self._check_params()

        was_series = isinstance(X, pd.Series)
        df = pd.DataFrame(X) if was_series else X

        n_samples = len(df)
        if n_samples > 0 and self.level > 0:
            max_level = max(1, int(math.log2(n_samples)))
            if self.level > max_level:
                raise ValueError(
                    f"`level` ({self.level}) exceeds maximum supported "
                    f"decomposition level ({max_level}) for signal length {n_samples}."
                )

        result = {}

        for col in df.columns:
            arr = df[col].to_numpy()
            n_orig = len(arr)
            nodes = self._decompose(arr, self.level)

            if self.output_feature == "energy":
                for i, coeffs in enumerate(nodes):
                    result[f"{col}_node_{i}_energy"] = [np.sum(coeffs**2)]

            elif self.output_feature == "entropy":
                for i, coeffs in enumerate(nodes):
                    power = coeffs**2
                    total = np.sum(power)
                    if total > 1e-12:
                        p = power / total
                        p = p[p > 1e-12]
                        entropy = -np.sum(p * np.log(p))
                    else:
                        entropy = 0.0
                    result[f"{col}_node_{i}_entropy"] = [entropy]

            elif self.output_feature == "coefficients":
                concat_coeffs = np.concatenate(nodes)
                result[col] = concat_coeffs[:n_orig]

        out = pd.DataFrame(result)

        if self.output_feature == "coefficients" and was_series:
            return pd.Series(out.iloc[:, 0], index=X.index, name=X.name)

        return out

    def _decompose(self, arr, level):
        """Recursively split signal into approximation and detail sub-bands.

        Uses the Haar wavelet: lowpass = (a + b) / sqrt(2),
        highpass = (a - b) / sqrt(2), applied pairwise.

        Returns a list of numpy arrays, one per terminal node.
        """
        if level == 0 or len(arr) < 2:
            return [arr]

        if len(arr) % 2 != 0:
            arr = np.pad(arr, (0, 1), mode="edge")

        half = len(arr) // 2
        if half == 0:
            return [arr]

        s = math.sqrt(2.0)
        approx = np.empty(half)
        detail = np.empty(half)
        for i in range(half):
            approx[i] = (arr[2 * i] + arr[2 * i + 1]) / s
            detail[i] = (arr[2 * i] - arr[2 * i + 1]) / s

        return self._decompose(approx, level - 1) + self._decompose(detail, level - 1)

    def _check_params(self):
        """Validate user-supplied parameters."""
        if not isinstance(self.level, int) or self.level < 0:
            raise ValueError(
                f"`level` must be a non-negative integer, got {self.level}"
            )
        valid = {"energy", "entropy", "coefficients"}
        if self.output_feature not in valid:
            raise ValueError(
                f"`output_feature` must be one of {valid}, got '{self.output_feature}'"
            )

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
        params1 = {"level": 2, "output_feature": "energy"}
        params2 = {"level": 1, "output_feature": "entropy"}
        params3 = {"level": 2, "output_feature": "coefficients"}
        return [params1, params2, params3]
