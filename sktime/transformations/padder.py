"""Padding transformer, pad unequal length panel to max length or fixed length."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__all__ = ["PaddingTransformer"]
__author__ = ["abostrom"]


class PaddingTransformer(BaseTransformer):
    """Padding panel of unequal length time series to equal, fixed length.

    Pads the input dataset to either an optional fixed length
    (longer than the longest series), or finds the max length series across
    all instances and pads to that with ``fill_value``.

    Parameters
    ----------
    pad_length : int, optional (default=None)
        Length to pad each series to. If ``None``, uses the longest series
        length observed in ``fit``.
    fill_value : float, optional (default=0)
        Value used for padding.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.transformations.padder import PaddingTransformer
    >>> X, _ = load_unit_test(return_X_y=True, split="train")
    >>> padder = PaddingTransformer(pad_length=30, fill_value=0.0)
    >>> Xt = padder.fit_transform(X)
    >>> len(Xt.iloc[0, 0])
    30
    """

    _tags = {
        "authors": ["abostrom"],
        "maintainers": ["abostrom"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "pd-multiindex",
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,
        "capability:unequal_length": True,
        "capability:unequal_length:removes": True,
        "capability:multivariate": True,
        "capability:categorical_in_X": False,
        # is transform result always guaranteed to be equal length (and series)?
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, pad_length=None, fill_value=0):
        self.pad_length = pad_length
        self.fill_value = fill_value
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame with pd.MultiIndex
            Panel data in ``pd-multiindex`` format. MultiIndex has two levels:
            first level is instance index, second level is time index.
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        if self.pad_length is None:
            self.pad_length_ = int(X.groupby(level=0).size().max())
        else:
            self.pad_length_ = self.pad_length

        return self

    def _pad_instance(self, Xi):
        """Pad a single instance DataFrame to ``pad_length_`` rows."""
        n = len(Xi)
        if n > self.pad_length_:
            raise ValueError(
                "Error: max_length of series "
                "is greater than the one found when fit or set."
            )
        if n == self.pad_length_:
            values = Xi.to_numpy(dtype=float, copy=True)
        else:
            values = np.full(
                (self.pad_length_, Xi.shape[1]), self.fill_value, dtype=float
            )
            values[:n] = Xi.to_numpy(dtype=float, copy=False)
        return pd.DataFrame(
            values, index=pd.RangeIndex(self.pad_length_), columns=Xi.columns
        )

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame with pd.MultiIndex
            Panel data in ``pd-multiindex`` format. MultiIndex has two levels:
            first level is instance index, second level is time index.
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame with pd.MultiIndex, same format as X
            transformed version of X with equal-length instances
        """
        if X.groupby(level=0).size().max() > self.pad_length_:
            raise ValueError(
                "Error: max_length of series "
                "is greater than the one found when fit or set."
            )

        pieces = []
        for inst, Xi in X.groupby(level=0, sort=False):
            Xi = Xi.droplevel(0)
            padded = self._pad_instance(Xi)
            padded.index = pd.MultiIndex.from_product([[inst], padded.index])
            pieces.append(padded)

        Xt = pd.concat(pieces, axis=0)
        Xt.index.names = X.index.names
        return Xt

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
            {"pad_length": None, "fill_value": 0},
            {"pad_length": 120, "fill_value": -1.0},
        ]
        return params
