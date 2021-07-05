#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova", "Markus Löning"]
__all__ = ["ColumnEnsembleTransformer"]

import numpy as np
import pandas as pd

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class ColumnEnsembleTransformer(_SeriesToSeriesTransformer):
    """Ensemble the original multivariate data into univariate pd.Series."""

    _tags = {
        "transform-returns-same-time-index": True,
        "univariate-only": False,
        "fit-in-transform": False,
    }

    def __init__(self, aggfunc="mean"):
        self.aggfunc = aggfunc
        super(ColumnEnsembleTransformer, self).__init__()

    def transform(self, Z, X=None):
        """Transform data.

        Parameters
        ----------
        Y : pd.DataFrame
            Multivariate series to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        column_ensemble: pd.Series
            Transformed univariate series.
        """
        valid_aggfuncs = {"mean": np.mean}
        if self.aggfunc not in valid_aggfuncs.keys():
            raise ValueError("Aggregation function %s not recognized." % self.aggfunc)

        column_ensemble = Z.apply(func=valid_aggfuncs[self.aggfunc], axis=1)
        return pd.Series(column_ensemble, index=Z.index)

    def _fit(self, Z, X=None):
        Z = check_series(Z)
        self._is_fitted = True
        return self
