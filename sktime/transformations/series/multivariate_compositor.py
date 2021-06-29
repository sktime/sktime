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
    columns : names of columns that are supposed to be transformed
    """

    _required_parameters = ["transformer"]
    _tags = {
        "multivariate-only": True,
    }

    def __init__(self, transformer, columns="all"):
        self.transformer = transformer
        self.columns = columns
        self.transformers_ = None
        super(MultivariateCompositor, self).__init__()

    def fit(self, Z, X=None):
        """
        Iterates over columns (series) and applies the fit function of the transformer
        """
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("Z needs to be a multivariate Pandas Series")
        self._is_fitted = False
        z = check_series(Z)
        if self.columns == "all":
            self.columns = z.columns

        # make sure z contains all columns that the user wants to transform
        Z_wanted_keys = set(self.columns)
        Z_new_keys = set(z.columns)
        difference = Z_wanted_keys.difference(Z_new_keys)
        if len(difference) != 0:
            raise ValueError("Missing columns" + str(difference) + "in Z.")

        self.transformers_ = {}
        for colname in self.columns:
            transformer = clone(self.transformer)
            self.transformers_[colname] = transformer
            self.transformers_[colname].fit(z[colname], X)
            # self.transformers_[colname].is_fitted = True
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """
        calls transform on every single transformer (one transformer per series)
        """
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("Z needs to be a multivariate Pandas Series")

        self.check_is_fitted()
        z = check_series(Z)

        # make copy of z
        z = z.copy()
        # make sure z contains all columns that the user wants to transform
        Z_wanted_keys = set(self.columns)
        Z_new_keys = set(z.columns)
        difference = Z_wanted_keys.difference(Z_new_keys)
        if len(difference) != 0:
            raise ValueError("Missing columns" + str(difference) + "in Z.")
        for colname in self.columns:
            # self.columns : columns that are supposed to be transformed
            self.transformers_[colname].check_is_fitted()
            z[colname] = self.transformers_[colname].transform(z[colname], X)
        return z

    def inverse_transform(self, Z, X=None):
        """
        if the base transformer has an inverse-transform this
        inverse transform is called on every single transformer
        (one transformer per series)
        """
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("Z needs to be a multivariate Pandas Series")

        if not hasattr(self.transformer, "inverse_transform"):
            raise NotImplementedError(
                "this transform does not have an inverse_transform method"
            )
        self.check_is_fitted()
        z = check_series(Z)

        if isinstance(Z, pd.DataFrame):
            # make copy of z
            z = z.copy()

            # make sure z contains all columns that the user wants to transform
            Z_wanted_keys = set(self.columns)
            Z_new_keys = set(z.columns)
            difference = Z_wanted_keys.difference(Z_new_keys)
            if len(difference) != 0:
                raise ValueError("Missing columns" + str(difference) + "in Z.")
            for colname in self.columns:
                # self.columns : columns that are supposed to be transformed
                self.transformers_[colname].check_is_fitted()
                z[colname] = self.transformers_[colname].inverse_transform(
                    z[colname], X
                )
            return z

    # todo: add functionality for update
