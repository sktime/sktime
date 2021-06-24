#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["MultivariateCompositor"]
__author__ = ["Svea Meyer"]

import pandas as pd
from sklearn.base import clone
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class MultivariateCompositor(_SeriesToSeriesTransformer):
    """
    Parameters
    ----------
    transformer : series-to-series transformer to be applied to each series

    """

    _required_parameters = ["transformer"]

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns
        self.transformers_ = None
        super(MultivariateCompositor, self).__init__()

    def fit(self, Z, X=None):
        """
        Iterates over columns (series) and applies the fit function of the transformer
        """
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("Z needs to be a multivariate series as a pd.DataFrame")
        self._is_fitted = False
        z = check_series(Z)
        if self.columns is None:
            self.columns = z.columns
        self.transformers_ = {}
        for colname in self.columns:
            transformer = clone(self.transformer)
            self.transformers_[colname] = transformer
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """
        calls transform on every single transformer (one transformer per series)
        """
        self.check_is_fitted()
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("Z needs to be a multivariate series as a pd.DataFrame")
        z = check_series(Z)
        for colname in self.columns:
            self.transformers_[colname].check_is_fitted()
            z[colname] = self.transformers_[colname].transform(z, X)
        return z

    def inverse_transform(self, Z, X=None):
        """
        if the base transformer has an inverse-transform this
        inverse transform is called on every single transformer
        (one transformer per series)
        """
        if not hasattr(self.transformer, "inverse_transform"):
            raise NotImplementedError(
                "this transform does not have an inverse_transform method"
            )
        self.check_is_fitted()
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("Z needs to be a multivariate series as a pd.DataFrame")
        z = check_series(Z)
        for colname in self.columns:
            self.transformers_[colname].check_is_fitted()
            z[colname] = self.transformers_[colname].inverse_transform(z, X)
        return z

    def apply(self, function, **kwargs):
        """
        apply any other function that is part of the transformer
        """
        for colname in self.columns:
            function(self.transformers_[colname], **kwargs)

        return self
