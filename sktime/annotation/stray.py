# -*- coding: utf-8 -*-
"""Tests for STRAY (Search TRace AnomalY) outlier estimator."""

import numpy as np
import pandas as pd
from scipy.stats import iqr
from sklearn.neighbors import NearestNeighbors

from sktime.annotation.base._base import BaseSeriesAnnotator

__author__ = ["KatieBuc"]
__all__ = ["STRAY"]


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

    def _find_threshold(outlier_score, alpha, p, tn, outtail):
        n = len(outlier_score)

        if outtail == "min":
            outlier_score = -outlier_score

        order = np.argsort(outlier_score)
        gaps = np.append(0, np.diff(outlier_score[order]))
        n4 = int(max(min(tn, np.floor(n / 4)), 2))

        J = np.array([i for i in range(2, n4 + 1)])
        start = int(max(np.floor(n * (1 - p)), 1))

        ghat = [
            0.0 if i < start else sum((J / (n4 - 1)) * gaps[i - J + 1])
            for i in range(n)
        ]

        log_alpha = np.log(1 / alpha)
        bound = np.Inf

        for i in range(start, n):
            if gaps[i] > log_alpha * ghat[i]:
                bound = outlier_score[order][i - 1]
                break

        return np.where(outlier_score > bound)

    def _use_KNN(X, alpha, k, knnsearchtype, p, tn, outtail):

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=knnsearchtype).fit(X)
        distances, indices = nbrs.kneighbors(X)
        if k == 1:
            d = distances[:, 1]
        else:
            diff = np.apply_along_axis(np.diff, 1, distances)
            d = distances[range(r), np.apply_along_axis(np.argmax, 1, diff) + 1]

        out_index = _find_threshold(d, alpha, p, tn, outtail)
        return {"idx_outliers": out_index, "out_scores": d}

    def _find_HDoutliers(
        X,
        alpha=0.01,
        k=10,
        knnsearchtype="brute",
        normalize="unitize",
        p=0.5,
        tn=50,
        outtail="max",
    ):
        r = np.shape(X)[0]
        idx_dropna = [i for i in range(r) if not np.isnan(X[i]).any()]  # tag
        X_dropna = X[
            idx_dropna,
        ]

        def unitize(x):
            diff = max(x) - min(x)
            if diff == 0:
                return np.zeros(len(x))
            return (x - min(x)) / diff

        def standardize(x):
            return (x - np.median(x)) / iqr(x)

        X_dropna = np.apply_along_axis(unitize, 0, X_dropna)
        outliers = _use_KNN(X_dropna, alpha, k, knnsearchtype, p, tn, outtail)

        idx_outliers = idx_dropna[outliers["idx_outliers"]]  # adjusted for missing data
        outlier_flag = [1 if i in idx_outliers else 0 for i in range(r)]

        return {
            "idx_outliers": idx_outliers,
            "out_scores": outliers["out_scores"],
            "outlier_flag": outlier_flag,
        }

    def _fit(self, X, Y=None):
        """Do nothing, currently empty.

        Parameters
        ----------
        X : 1D np.array, shape = [num_observations]
            Observations to apply labels to.

        Returns
        -------
        self :
            Reference to self.
        """
        return self

    def _predict(self, X):
        """Something."""
