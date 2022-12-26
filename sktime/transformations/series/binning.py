# -*- coding: utf-8 -*-
"""Time binning for turning series equally spaced."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import warnings

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class TimeBinAggregate(BaseTransformer):
    r"""Bins time series and aggregates by bin.

    In `transform`, applies `groupby` with `aggfunc` on the temporal coordinate.

    More precisely:
    `bins` encodes bins :math:`B_1, \dots, B_k` where :math:`B_i` are intervals,
    in the reals or in a temporal (time stamp) range.

    In `transform`, the estimator `TimeBinAggregate` collects values
    at time stamps of `X` falling into :math:`B_i` as a sample :math:`S_i`,
    and then applies `aggfunc` to :math:`S_i` to obtain an aggregate value :math:`v_i`.
    The transformed series are values :math:`v_i` at time stamps :math:`t_i`,
    determined from :math:`B_i` per the rule in `return_index`.

    Parameters
    ----------
    bins : 1D array-like or pd.IntervalIndex
        if 1D array-like, is interpreted as breaks of bins
        breaks of bins defining intervals considered by aggfunc
    aggfunc : callable *1D array-like -> float), optional, default=np.mean
        Function used to aggregate the values in intervals.
        Should have signature 1D -> float and defaults
        to mean if None
    return_index : str, one of the below; optional, default="bin_start"
        "bin_start" = transformed pd.DataFrame will be indexed by bin starts
        "bin_end" = transformed pd.DataFrame will be indexed by bin starts
        "bin_mid" = transformed pd.DataFrame will be indexed by bin midpoints
        "bin" = transformed pd.DataFrame will have `bins` as `IntervalIndex`

    Example
    -------
    from sktime.datatypes import get_examples
    from sktime.transformations.series.binning import TimeBinAggregate

    bins = [0, 2, 4]
    X = get_examples("pd.DataFrame")[0]

    t = TimeBinAggregate([-1, 2, 10])
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # and for y?
    }

    def __init__(self, bins, aggfunc=None, return_index="bin_start"):

        self.bins = bins
        self.aggfunc = aggfunc
        self.return_index = return_index

        if not isinstance(bins, pd.IntervalIndex):
            self._bins = pd.IntervalIndex.from_breaks(bins)
        else:
            self._bins = bins

        if self.aggfunc is None:
            self._aggfunc = np.mean
        else:
            assert callable(aggfunc), (
                "aggfunc should be callable with" "signature 1D -> float"
            )
            if aggfunc.__name__ == "<lambda>":
                warnings.warn("Save and load will not work with lambda functions")
            self._aggfunc = self.aggfunc

        super(TimeBinAggregate, self).__init__()

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
        idx_cut = pd.cut(X.index, bins=self._bins, include_lowest=True)
        Xt = X.groupby(idx_cut).apply(self._aggfunc)

        if self.return_index == "bin_start":
            Xt.index = [x.left for x in Xt.index]
        elif self.return_index == "bin_end":
            Xt.index = [x.right for x in Xt.index]
        elif self.return_index == "bin_mid":
            Xt.index = [(x.left + x.right) / 2 for x in Xt.index]
        elif self.return_index == "bin":
            Xt.index = self._bins
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
        bins = pd.interval_range(start=0, end=100, freq=10, closed="left")
        params1 = {"bins": bins}

        params2 = {"bins": [0, 2, 4], "aggfunc": np.sum, "return_index": "bin_mid"}
        return [params1, params2]
