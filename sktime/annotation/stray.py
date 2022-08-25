# -*- coding: utf-8 -*-
"""Tests for STRAY (Search TRace AnomalY) outlier estimator."""

import numpy as np
from scipy.stats import iqr
from sklearn.neighbors import NearestNeighbors

from sktime.annotation.base._base import BaseSeriesAnnotator

__author__ = ["KatieBuc"]
__all__ = ["STRAY"]


def unitize(x):
    diff = max(x) - min(x)
    if diff == 0:
        return np.zeros(len(x))
    return (x - min(x)) / diff


def standardize(x):
    return (x - np.median(x)) / iqr(x)


class STRAY(BaseSeriesAnnotator):
    """
    Description.

    Parameters
    ----------
    ...

    Examples
    --------
    ...
    """

    def __init__(
        self,
        alpha=0.01,
        k=10,
        knnsearchtype="brute",
        normalize=unitize,
        p=0.5,
        tn=50,
        outtail="max",
        labels="indicator",
    ):
        self.alpha = alpha
        self.k = k
        self.knnsearchtype = knnsearchtype
        self.normalize = normalize
        self.p = p
        self.tn = tn
        self.outtail = outtail
        self.labels = labels
        super(STRAY, self).__init__(fmt="dense", labels=self.labels)

    def _find_threshold(self, outlier_score, n):

        if self.outtail == "min":
            outlier_score = -outlier_score

        order = np.argsort(outlier_score)
        gaps = np.append(0, np.diff(outlier_score[order]))
        n4 = int(max(min(self.tn, np.floor(n / 4)), 2))

        J = np.array([i for i in range(2, n4 + 1)])
        start = int(max(np.floor(n * (1 - self.p)), 1))

        ghat = [
            0.0 if i < start else sum((J / (n4 - 1)) * gaps[i - J + 1])
            for i in range(n)
        ]

        log_alpha = np.log(1 / self.alpha)
        bound = np.Inf

        for i in range(start, n):
            if gaps[i] > log_alpha * ghat[i]:
                bound = outlier_score[order][i - 1]
                break

        return np.where(outlier_score > bound)[0]

    def _use_KNN(self, X, n):

        if len(X.shape) == 1:
            nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X.reshape(-1, 1))
            distances, _ = nbrs.kneighbors(X.reshape(-1, 1))
        else:
            nbrs = NearestNeighbors(
                n_neighbors=self.k + 1, algorithm=self.knnsearchtype
            ).fit(X)
            distances, _ = nbrs.kneighbors(X)

        if self.k == 1:
            d = distances[:, 1]
        else:
            diff = np.apply_along_axis(np.diff, 1, distances)
            d = distances[range(n), np.apply_along_axis(np.argmax, 1, diff) + 1]

        out_index = self._find_threshold(d, n)
        return {"idx_outliers": out_index, "out_scores": d}

    def _find_HDoutliers(self, X):
        r = np.shape(X)[0]
        idx_dropna = np.array([i for i in range(r) if not np.isnan(X[i]).any()])  # tag
        X_dropna = X[
            idx_dropna,
        ]

        X_dropna = np.apply_along_axis(self.normalize, 0, X_dropna)

        n = np.shape(X_dropna)[0]
        outliers = self._use_KNN(X_dropna, n)

        # adjusted back to length r, for missing data
        slice_ = [True if i in outliers["idx_outliers"] else False for i in range(n)]
        idx_outliers = idx_dropna[slice_]
        outlier_bool = [1 if i in idx_outliers else 0 for i in range(r)]
        outlier_scores = [
            np.nan if i in idx_outliers else outliers["out_scores"][i] for i in range(r)
        ]

        return {
            "outlier_scores": outlier_scores,
            "outlier_bool": outlier_bool,
        }

    def _fit(self, X, Y=None):
        """Do nothing, currently empty.

        Parameters
        ----------
        ...

        Returns
        -------
        self :
            Reference to self.
        """
        return self

    def _predict(self, X):
        """Something."""

        info_dict = self._find_HDoutliers(X)

        if self.labels == "scores":
            return info_dict["outlier_scores"]
        else:
            return info_dict["outlier_bool"]
