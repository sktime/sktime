# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator with fixed parameters."""

__author__ = ["fkiraly"]
__all__ = ["FixedParams"]

from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.param_est.base import BaseParamFitter


class FixedParams(BaseParamFitter):
    """Dummy parameter estimator that writes fixed values to self.

    This can be used as a dummy/mock, or as a pipeline element, e.g.,
    to set parameters to certain values, or in model selection as the "fixed" option.

    Takes a dictionary `param_dict` of name/value pairs to write to self in `fit`.
    In `fit`, for each key-value pair in `param_dict`,
    writes `value` to attribute `str(key) + "_"` in `self`

    Parameters
    ----------
    param_dict : dict
        fixed parameter values written to `self`
    """

    _tags = {
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        # which types do _fit/_predict, support for X?
        "scitype:X": ["Series", "Panel", "Hierarchical"],
        # which X scitypes are supported natively?
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
    }

    def __init__(self, param_dict):
        self.param_dict = param_dict
        super(FixedParams, self).__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        param_dict = self.param_dict

        if isinstance(param_dict, dict):
            for key, value in param_dict.items():
                setattr(self, f"{str(key)}_", value)

        return self

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
        params1 = {"param_dict": {1: 2}}
        params2 = {"param_dict": {"foo": "bar", "bar": "foo"}}

        return [params1, params2]
