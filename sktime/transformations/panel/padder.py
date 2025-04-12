"""Padding transformer, pad unequal length panel to max length or fixed length."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.pandas import df_map

__all__ = ["PaddingTransformer"]
__author__ = ["abostrom"]


class PaddingTransformer(BaseTransformer):
    """Padding panel of unequal length time series to equal, fixed length.

    Pads the input dataset to either a optional fixed length
    (longer than the longest series).
    Or finds the max length series across all series and dimensions and
    pads to that with zeroes.

    Parameters
    ----------
    pad_length : int, optional (default=None) length to pad the series too.
        if None, will find the longest sequence and use instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.panel.padder import PaddingTransformer
    >>>
    >>> # Create a sample nested DataFrame with unequal length time series
    >>> data = {
    ...     'feature1': [
    ...         pd.Series([1, 2, 3]), pd.Series([4, 5]), pd.Series([6, 7, 8, 9])
    ...     ],
    ...     'feature2': [
    ...         pd.Series([10, 11]), pd.Series([12, 13, 14]), pd.Series([15])
    ...     ]
    ... }
    >>> X = pd.DataFrame(data)
    >>>
    >>> # Initialize the PaddingTransformer
    >>> padder = PaddingTransformer()
    >>>
    >>> # Fit the transformer to the data
    >>> padder.fit(X)
    PaddingTransformer()
    >>>
    >>> # Transform the data
    >>> Xt = padder.transform(X)
    >>>
    >>> # Display the transformed data
    >>> # print(Xt)
    """

    _tags = {
        "authors": ["abostrom"],
        "maintainers": ["abostrom"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
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
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        if self.pad_length is None:
            n_instances, _ = X.shape
            arr = [X.iloc[i, :].values for i in range(n_instances)]
            self.pad_length_ = _get_max_length(arr)
        else:
            self.pad_length_ = self.pad_length

        return self

    def _create_pad(self, series):
        out = np.full(self.pad_length_, self.fill_value, float)
        out[: len(series)] = series.iloc[: len(series)]
        return out

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        n_instances, _ = X.shape

        arr = [X.iloc[i, :].values for i in range(n_instances)]

        max_length = _get_max_length(arr)

        if max_length > self.pad_length_:
            raise ValueError(
                "Error: max_length of series \
                    is greater than the one found when fit or set."
            )

        pad = [pd.Series([self._create_pad(series) for series in out]) for out in arr]
        Xt = df_map(pd.DataFrame(pad))(pd.Series)
        Xt.columns = X.columns

        return Xt


def _get_max_length(X):
    def get_length(input):
        return max(map(lambda series: len(series), input))

    return max(map(get_length, X))
