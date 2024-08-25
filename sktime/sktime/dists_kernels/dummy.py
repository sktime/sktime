"""Dummy distances/kernels."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels.base import BasePairwiseTransformerPanel

SUPPORTED_MTYPES = ["df-list", "nested_univ", "numpy3D"]


class ConstantPwTrafoPanel(BasePairwiseTransformerPanel):
    """Pairwise panel transformer that always returns a constant value.

    Parameters
    ----------
    constant : float, optional, default = 0
        the constant value that this transformer returns
    """

    _tags = {
        "authors": "fkiraly",
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
    }

    def __init__(self, constant=0):
        self.constant = constant

        super().__init__()

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from public transform

        Parameters
        ----------
        X: sktime Panel data container
        X2: sktime Panel data container

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        # get metadata from self - this is not always len(X) etc
        #   but for the mtypes allowed, it is
        # todo: conversion is wasteful, replace this perhaps by base class functionality
        #   we check the input type anyway, so we could store n_instances
        #   (but the problem there is, we should not write to self in transform)
        n = len(X)
        if X2 is not None:
            m = len(X2)
        else:
            m = n

        distmat = self.constant * np.ones((n, m))

        return distmat

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        params1 = {}
        params2 = {"constant": 42}

        return [params1, params2]
