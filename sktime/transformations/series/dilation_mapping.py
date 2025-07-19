"""DilationMapping transformer.

A transformer for applying dilation mapping to time series data.
"""

__author__ = ["fspinna"]
__all__ = ["DilationMappingTransformer"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class DilationMappingTransformer(BaseTransformer):
    r"""Dilation mapping transformer.

    A transformer for applying an index grid dilation mapping to time series data,
    in the terminology of [1]_.

    This transformation is motivated by kernel dilation, it
    reorders the timesteps of a time series to simulate the effect of dilation.
    For instance, in a pipeline, it enables a dilation-like effect for downstream
    models that do not inherently support such a feature.

    Mathematically, the mapping operates on sequences :math:`x_1, \dots, x_k`.
    The dilation with factor :math:`d` is defined as the sequence
    :math:`x_1, x_{1+d}, x_{1+2d}, \dots, x_2, x_{2+d}, x_{2+2d}, \dots, x_d, x_{2d},
    \dots`,
    where the subsequences with grid spacing :math:`d` are maximal.

    The resulting sequence is of equal length to the input sequence.

    This transformer reorders the values, and resets the sequence index
    to a ``RangeIndex``, if the mtype is ``pandas`` based.

    Parameters
    ----------
    dilation : int, default=2
        The dilation factor. Determines the spacing between original data points in the
        transformed series. Must be an integer greater than 0. A dilation of 1 means no
        change, while higher values increase the spacing.

    References
    ----------
    .. [1] Patrick Schäfer and Ulf Leser,
       "WEASEL 2.0--A Random Dilated Dictionary Transform for Fast,
       Accurate and Memory Constrained Time Series Classification", 2023,
       arXiv preprint arXiv:2301.10194.

    Examples
    --------
    >>> from sktime.transformations.series.dilation_mapping import \
    ...     DilationMappingTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> y_transform = DilationMappingTransformer(dilation=2).fit_transform(y)
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["fspinna"],
        "maintainers": ["fspinna"],
        # estimator type
        # --------------
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
        "capability:missing_values": True,
    }

    def __init__(self, dilation=2):
        self.dilation = dilation

        super().__init__()

        if dilation < 1:
            raise ValueError("Dilation must be greater than 0")

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        #  X_transformed : Series of mtype pd.Series
        #       transformed version of X
        """
        return self._dilate_series(X, self.dilation)

    def _dilate_series(self, x, d):
        x_dilations = [x[i::d] for i in range(0, d)]
        x_dilated = pd.concat(x_dilations, axis=0)
        x_dilated.name = x.name
        return x_dilated.reset_index(drop=True)

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
        params = [{"dilation": 2}]
        return params
