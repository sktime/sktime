"""Symbolic Aggregate Approximation Transformer."""

__author__ = ["steenrotsman"]
import numpy as np
from scipy.stats import norm, zscore

from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.paa import PAA


class SAX(BaseTransformer):
    """Symbolic Aggregate approXimation Transformer (SAX).

    SAX [2]_ is a dimensionality reduction technique that z-normalises a time
    series, applies Piecewise Aggregate Approximation (PAA) [1]_, and bins the
    mean of each PAA frame to a discrete value, resulting in a SAX word.

    This implementation offers two variants:
     1) the original, which takes the desired number of frames and can set the
     frame size to a fraction to support cases where the time series cannot be
     divided into the frames equally.
     2) a variant that takes the desired frame size and can decrease the frame
     size of the last frame to support cases where the time series is not
     evenly divisible into frames.

    Parameters
    ----------
    word_size : int, optional (default=8, greater equal 1 if frame_size=0)
        length of transformed time series. Ignored if ``frame_size`` is set.
    alphabet_size : int, optional (default=5, greater equal 2)
        number of discrete values transformed time series is binned to.
    frame_size : int, optional (default=0, greater equal 0)
        length of the frames over which the mean is taken. Overrides ``frames`` if > 0.

    References
    ----------
    .. [1] Keogh, E., Chakrabarti, K., Pazzani, M., and Mehrotra, S.
        Dimensionality Reduction for Fast Similarity Search
        in Large Time Series Databases.
        Knowledge and Information Systems 3, 263–286 (2001).
        https://doi.org/10.1007/PL00011669
    .. [2] Lin, J., Keogh, E., Wei, L., and Lonardi, S.
        Experiencing SAX: A Novel Symbolic Representation of Time Series.
        Data Mining and Knowledge Discovery 15, 107–144 (2007).
        https://doi.org/10.1007/s10618-007-0064-z

    Examples
    --------
    >>> from numpy import arange
    >>> from sktime.transformations.series.sax import SAX

    >>> X = arange(10)
    >>> sax = SAX(word_size=3, alphabet_size=5)
    >>> sax.fit_transform(X)  # doctest: +SKIP
    array([0, 2, 4])
    >>> sax = SAX(frame_size=2, alphabet_size=5)  # doctest: +SKIP
    array([0, 1, 2, 3, 4])
    """

    _tags = {
        "authors": ["steenrotsman"],
        "maintainers": ["steenrotsman"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "np.ndarray",
        "y_inner_mtype": "None",
        "univariate-only": True,
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "handles-missing-data": False,
    }

    def __init__(self, word_size=8, alphabet_size=5, frame_size=0):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.frame_size = frame_size

        super().__init__()

        self._check_params()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series of mtype np.ndarray
            data to be transformed
        y : None
            unused

        Returns
        -------
        X_transformed : Series of mtype np.ndarray
            transformed version of X
        """
        X_transformed = zscore(X)
        paa = PAA(self.word_size, self.frame_size)
        X_transformed = paa.fit_transform(X_transformed)
        X_transformed = np.digitize(X_transformed, self._get_breakpoints())
        return X_transformed

    def _get_breakpoints(self):
        return norm.ppf(np.arange(1, self.alphabet_size) / self.alphabet_size, loc=0)

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
        params = {"word_size": 4, "alphabet_size": 5}
        return params

    def _check_params(self):
        for attribute in ["word_size", "alphabet_size", "frame_size"]:
            if not isinstance(getattr(self, attribute), int):
                t = type(getattr(self, attribute)).__name__
                raise TypeError(f"{attribute} must be of type int. Found {t}.")

        if self.word_size < 1 and not self.frame_size:
            raise ValueError("word_size must be at least 1.")
        if self.alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2.")
        if self.frame_size < 0:
            raise ValueError("frame_size must be at least 0.")
