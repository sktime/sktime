# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Simple repetition bootstrap.."""

__author__ = ["fkiraly"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class RepeatBootstrapTransformer(BaseTransformer):
    """Repetition bootstrap, repeats given time series identically, ``n_series`` times.

    Useful as a baseline method, or for random reinitialization of a model.

    Parameters
    ----------
    n_series : int, optional
        The number of repeats that will be generated, by default 10.
    """

    _tags = {
        "authors": "fkiraly",
        # todo: what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # todo: what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # todo: what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "capability:missing_values": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
    }

    def __init__(self, n_series=10):
        self.n_series = n_series

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : ignored, for interface compatibility

        Returns
        -------
        transformed version of X
        """
        n_repeats = self.n_series

        df_list = [X for _ in range(n_repeats)]
        Xt = pd.concat(df_list, keys=range(n_repeats))

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{}, {"n_series": 1}, {"n_series": 3}]

        return params
