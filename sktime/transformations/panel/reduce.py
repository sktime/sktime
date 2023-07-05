"""Tabularizer transform, for pipelining."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly", "kcc-lion"]
__all__ = ["Tabularizer"]

import warnings

import numpy as np
import pandas as pd

from sktime.datatypes import convert, convert_to
from sktime.transformations.base import BaseTransformer


class Tabularizer(BaseTransformer):
    """A transformer that turns time series/panel data into tabular data.

    This estimator converts nested pandas dataframe containing time-series/panel data
    with numpy arrays or pandas Series in dataframe cells into a tabular pandas
    dataframe with only primitives in cells. This is useful for transforming time-
    series/panel data into a format that is accepted by standard validation learning
    algorithms (as in sklearn).
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["nested_univ", "numpy3D"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # and for y?
    }

    def _transform(self, X, y=None):
        """Transform nested pandas dataframe into tabular dataframe.

        Parameters
        ----------
        X : pandas DataFrame or 3D np.ndarray
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with only primitives in cells.
        """
        Xt = convert_to(X, to_type="numpyflat", as_scitype="Panel")
        return Xt

    def inverse_transform(self, X, y=None):
        """Transform tabular pandas dataframe into nested dataframe.

        Parameters
        ----------
        X : pandas DataFrame
            Tabular dataframe with primitives in cells.
        y : array-like, optional (default=None)

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with series in cells.
        """
        Xt = convert(X, from_type="numpyflat", to_type="numpy3D", as_scitype="Panel")
        return Xt


class TimeBinner(BaseTransformer):
    """Turns time series/panel data into tabular data based on intervals.

    This estimator converts nested pandas dataframe containing
    time-series/panel data with numpy arrays or pandas Series in
    dataframe cells into a tabular pandas dataframe with only primitives in
    cells. The primitives are calculated based on Intervals defined
    by the IntervalIndex and aggregated by aggfunc.

    This is useful for transforming time-series/panel data
    into a format that is accepted by standard validation learning
    algorithms (as in sklearn).

    Parameters
    ----------
    idx : pd.IntervalIndex
        IntervalIndex defining intervals considered by aggfunc
    aggfunc : callable
        Function used to aggregate the values in intervals.
        Should have signature 1D -> float and defaults
        to mean if None
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["nested_univ"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # and for y?
    }

    def __init__(self, idx, aggfunc=None):
        assert isinstance(
            idx, pd.IntervalIndex
        ), "idx should be of type pd.IntervalIndex"
        self.aggfunc = aggfunc
        if self.aggfunc is None:
            self._aggfunc = np.mean
            warnings.warn(
                "No aggfunc was passed, defaulting to mean",
                stacklevel=2,
            )
        else:
            assert callable(aggfunc), (
                "aggfunc should be callable with" "signature 1D -> float"
            )
            if aggfunc.__name__ == "<lambda>":
                warnings.warn(
                    "Save and load will not work with lambda functions",
                    stacklevel=2,
                )
            self._aggfunc = self.aggfunc
        self.idx = idx

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        idx = pd.cut(X.iloc[0, 0].index, bins=self.idx, include_lowest=True)
        Xt = X.applymap(lambda x: x.groupby(idx).apply(self._aggfunc))
        Xt = convert_to(Xt, to_type="numpyflat", as_scitype="Panel")
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        import pandas as pd

        idx = pd.interval_range(start=0, end=100, freq=10, closed="left")
        params = {"idx": idx}
        return params
