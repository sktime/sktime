#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Leonidas Tsaprounis"]
__all__ = ["tsfeatures"]

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.panel import check_X
import pandas as pd
import numpy as np

_check_soft_dependencies("tsfeatures")
from tsfeatures.tsfeatures import tsfeatures  # noqa: E402


class TSFeatures(_PanelToTabularTransformer):
    """
    Adapter for tsfeatures: https://github.com/FedericoGarza/tsfeatures
    python tsfeatures authors:
        Federico Garza - FedericoGarza
        Kin Gutierrez - kdgutier
        Cristian Challu - cristianchallu
        Jose Moralez - jose-moralez
        Ricardo Olivares - rolivaresar
        Max Mergenthaler - mergenthaler
    """

    def __init__(self, a):
        self.a = a

    def transform(self, X, y=None):
        """
        Transforms data into tsfeatures feeatures

        Parameters
        ----------
        X : pandas DataFrame, input time series
        y : array_like, target values (optional, ignored)

        Returns
        -------
        Pandas dataframe containing 22 features for each input series
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        return pd.DataFrame(X)


feature_names = (
    "acf_features",
    "heterogeneity",
    "series_length",
    "arch_stat",
    "holt_parameters",
    "sparsity",
    "count_entropy",
    "hurst",
    "stability",
    "crossing_points",
    "hw_parameters",
    "stl_features",
    "entropy",
    "intervals",
    "unitroot_kpss",
    "flat_spots",
    "lumpiness",
    "unitroot_pp",
    "frequency",
    "nonlinearity",
    "guerrero",
    "pacf_features",
)
