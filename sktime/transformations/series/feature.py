#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements feature selection algorithms."""

__author__ = ["aiwalter"]
__all__ = ["FeatureSelection", "Featureizer"]

from sklearn.base import clone

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.transformations.series.feature_selection import FeatureSelection


class Featureizer(_SeriesToSeriesTransformer):
    """Create new exogenous features based on a given transformer.

    Parameters
    ----------
    transformer: Tuple of an sktime-like transformer ("name", class) or list of such tuples.
        The given transformer(s) receive the given X as input and append the output
        to Z as a new column with the transformer name as suffix.

    Attributes
    ----------
        transformer_

    Examples
    --------

    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        "univariate-only": True,
    }

    def __init__(
        self,
        transformer,
        fh,
        temporal_feature=False,
        suffix=None,
    ):
        self.transformer = transformer
        self.fh = fh
        self.temporal_feature = temporal_feature
        self.suffix = suffix

        super(Featureizer, self).__init__()

    def fit(self, Z, X=None):
        """Fit the transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.DataFrame
            A time series to apply the transformation on.
        X : pd.pd.Series, default=None
            Featureizer needs the target series y given as X in order to
            return Z with the newly added feature.

        Returns
        -------
        self
        """
        Z = check_series(Z)
        self._check_transformer()
        self.transformer_ = clone(self.transformer)
        if not self.temporal_feature:
            # swap Z and X
            self.transformer_.fit(Z=X, X=Z)
        if self.suffix is None:
            self.suffix =  "_" + self.transformer.__class__.__name__.lower()

        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame
            A time series to apply the transformation on.
        X : pd.DataFrame, default=None
            Exogenous data is ignored in transform.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        Zt = Z.copy()
        Zt[self.suffix] = self.transformer_.transform()
        return Zt

    def _check_transformer(self):
        valid_transformer_type = _SeriesToSeriesTransformer
        for t in self.transformer:
            if not isinstance(t, valid_transformer_type):
                raise TypeError(
                    f"All intermediate steps should be "
                    f"instances of {valid_transformer_type}, "
                    f"but transformer: {t} is not."
                )
