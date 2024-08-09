"""Piecewise Aggregate Approximation Transformer."""

__author__ = ["steenrotsman"]

import numpy as np

from sktime.transformations.base import BaseTransformer


class PAA(BaseTransformer):
    """Piecewise Aggregate Approximation Transformer (PAA).

    PAA [1]_ is a dimensionality reduction technique that divides a time series
    into frames and takes their mean. This implementation offers two variants:
     1) the original, which takes the desired number of frames and can set the
     frame size to a fraction to support cases where the time series cannot be
     divided into the frames equally.
     2) a variant that takes the desired frame size and can decrease the frame
     size of the last frame to support cases where the time series is not
     evenly divisible into frames.

    Parameters
    ----------
    frames : int, optional (default=8, greater equal 1 if frame_size=0)
        length of transformed time series. Ignored if ``frame_size`` is set.
    frame_size : int, optional (default=0, greater equal 0)
        length of the frames over which the mean is taken. Overrides ``frames`` if > 0.

    References
    ----------
    .. [1] Keogh, E., Chakrabarti, K., Pazzani, M., and Mehrotra, S.
        Dimensionality Reduction for Fast Similarity Search
        in Large Time Series Databases.
        Knowledge and Information Systems 3, 263-286 (2001).
        https://doi.org/10.1007/PL00011669

    Examples
    --------
    >>> from numpy import arange
    >>> from sktime.transformations.series.paa import PAA

    >>> X = arange(10)
    >>> paa = PAA(frames=3)
    >>> paa.fit_transform(X)  # doctest: +SKIP
    array([1.2, 4.5, 7.8])
    >>> paa = PAA(frame_size=3)  # doctest: +SKIP
    array([1, 4, 7, 9])
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

    def __init__(self, frames=8, frame_size=0):
        self.frames = frames
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
        if self.frame_size:
            return self._transform_frame_size(X)
        return self._transform_frames(X)

    def _transform_frames(self, X):
        """Original PAA definition.

        First performs validity check and handles trivial cases. If self.frames evenly
        divides X, calculation is easy. Else, values in X are weighed into the frames.
        Last case corrects and adapts https://vigne.sh/posts/piecewise-aggregate-approx/
        """
        if self.frames > X.shape[0]:
            raise ValueError(
                "Series length cannot be shorter than the desired number of frames."
            )
        elif not X.shape[0] % self.frames:
            return X.reshape(self.frames, -1).mean(axis=1).T

        indices = np.arange(X.shape[0] * self.frames) // self.frames
        return X[indices.reshape(self.frames, -1)].sum(axis=1) / X.shape[0]

    def _transform_frame_size(self, X):
        if self.frame_size > X.shape[0]:
            raise ValueError(
                "Series length cannot be shorter than the desired frame size."
            )
        elif last_frame_length := (X.shape[0] % self.frame_size):
            last_frame_mean = np.mean(X[-last_frame_length:])
            last_frame_fill = [last_frame_mean] * (self.frame_size - last_frame_length)
            X = np.append(X, last_frame_fill)

        return np.mean(X.reshape(-1, self.frame_size), axis=1).T

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
        params = {"frames": 4}
        return params

    def _check_params(self):
        for attribute in ["frames", "frame_size"]:
            if not isinstance(getattr(self, attribute), int):
                t = type(getattr(self, attribute)).__name__
                raise TypeError(f"{attribute} must be of type int. Found {t}.")

        if self.frames < 1 and not self.frame_size:
            raise ValueError("frames must be at least 1.")
