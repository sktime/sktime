# -*- coding: utf-8 -*-
"""Preprocessing algorithm DOBIN (Distance based Outlier BasIs using Neighbors)."""

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.linalg import null_space
from scipy.stats import iqr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from sktime.transformations.base import BaseTransformer

__author__ = ["KatieBuc"]
__all__ = ["DOBIN"]


def unitize(x):
    diff = max(x) - min(x)
    if diff == 0:
        return np.zeros(len(x))
    return (x - min(x)) / diff


def standardize_median(x):
    if iqr(x) == 0:
        return x
    return (x - np.median(x)) / iqr(x)


def standardize_mean(x):
    if np.std(x) == 0:
        return x
    return (x - np.mean(x)) / np.std(x)


def identity(x):
    return x


class DOBIN(BaseTransformer):
    """Distance based Outlier BasIs using Neighbors (DOBIN).

    DOBIN is a pre-processing algorithm that constructs a set of basis
    basistors tailored for outlier detection as described by _[1]. DOBIN
    has a simple mathematical foundation and can be used as a dimension
    reduction tool for outlier detection tasks.

    Parameters
    ----------
    frac : int, float (default=0.95)
        The cut-off quantile for Y space
    normalize : callable {unitize, standardize_median, standardize_mean}
        (default=unitize)
        Method to normalize the columns of the data. This prevents variables
        with large variances having disproportional influence on Euclidean distances.
    k : int (default=None)
        Number of nearest neighbours considered, with a default value None calculated
        as 5% of the number of observations with a cap of 20.

    Attributes
    ----------
    _basis
    _coords

    References
    ----------
    [1] Kandanaarachchi, Sevvandi, and Rob J. Hyndman. "Dimension reduction
    for outlier detection using DOBIN." Journal of Computational and Graphical
    Statistics 30.1 (2021): 204-219.

    Examples
    --------
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
        "skip-inverse-transform": True,
    }

    def __init__(
        self,
        frac=0.95,
        normalize=unitize,  # TODO: make it so that we accept sktime normalizations
        k=None,
    ):
        self.frac = frac
        self.normalize = normalize
        self.k = k
        super(DOBIN, self).__init__()

    def _fit(self, X, y=None):

        self._X = X
        X = X.apply(self.normalize, axis=0)

        assert all(X.apply(is_numeric_dtype, axis=0))

        n_obs, n_dim = X.shape

        if n_obs < n_dim:
            # more dimensions than observations, change of basis
            # to subspace
            pca = PCA(n_components=n_obs)
            X = pca.fit_transform(X)
            _, n_dim = X.shape

        if self.k is None:
            self.k = min(20, max(n_obs // 20, 2))

        X_copy = X.copy()
        B = np.identity(n_dim)
        basis = pd.DataFrame()

        for _ in range(n_dim):
            # Compute Y space
            y_space = close_distance_matrix(X_copy, self.k, self.frac)

            # Find eta
            w = y_space.apply(sum, axis=0)
            eta = np.array([w / np.sqrt(sum(w**2))])

            # Update basis
            basis_col = pd.DataFrame(np.dot(B, eta.T))
            basis = pd.concat([basis, basis_col], axis=1)

            # Find xperp
            xperp = X_copy - np.dot(np.dot(np.array(X_copy), eta.T), eta)

            # Find a basis B for xperp
            B1 = null_space(eta)

            # Change xperp coordinates to B basis
            X_copy = np.dot(xperp, B1)

            # Update B with B1, each time 1 dimension is reduced
            B = np.dot(B, B1)

        # new coordinates
        coords = X.dot(basis)

        self._basis = basis
        self._coords = coords

        return self


def _transform(self, X, y=None):

    # fit again if indices not seen, but don't store anything
    if not X.index.equals(self._X.index):
        X_full = X.combine_first(self._X)
        new_dobin = DOBIN(
            frac=self.frac,
            normalize=self.normalize,
            k=self.k,
        ).fit(X_full)
        warnings.warn(
            "Warning: Input data X differs from that given to fit(). "
            "Refitting with new input data, not storing updated public class "
            "attributes. For this, explicitly use fit(X) or fit_transform(X)."
        )
        return new_dobin._coords

    return self._coords


def close_distance_matrix(X, k, frac):

    X = pd.DataFrame(X)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(
        X
    )  # FIXME: what default distance metric??
    _, indices = nbrs.kneighbors(X)

    dist = pd.DataFrame(
        [
            (
                (
                    X.loc[
                        i,
                    ]
                    - X.loc[
                        j,
                    ]
                )
                ** 2
            ).tolist()
            for (i, j) in zip(
                np.repeat(indices[:, 0], repeats=k), indices[:, 1:].flatten()
            )
        ]
    )

    row_sums = dist.apply(sum, axis=1)
    q_frac = np.quantile(row_sums, q=frac)

    mask = row_sums > q_frac

    return dist.loc[
        mask,
    ]
