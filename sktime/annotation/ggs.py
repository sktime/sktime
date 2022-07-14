# -*- coding: utf-8 -*-
"""
Greedy Gaussian Segmentation (GGS).

Based on:

    Hallac, D., Nystrup, P. & Boyd, S.
    Greedy Gaussian segmentation of multivariate time series.
    Adv Data Anal Classif 13, 727–751 (2019).
    https://doi.org/10.1007/s11634-018-0335-0

- source code adapted based on: https://github.com/cvxgrp/GGS
- paper available at: https://stanford.edu/~boyd/papers/pdf/ggs.pdf
"""

import abc
import logging
import math
import random
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GGS:
    """
    Greedy Gaussian Segmentation.

    Args
    ----
        k_max: maximum number of change points to find
        lamb: regularization parameter
        max_shuffles:
        verbose:
    """

    def __init__(
        self, k_max: int, lamb: float, max_shuffles: int = 250, verbose: bool = False
    ):
        self.k_max = k_max
        self.lamb = lamb
        self.max_shuffles = max_shuffles
        self.verbose = verbose

    def initialize_intermediates(self):
        """Fit."""
        self.intermediate_change_points = []
        self.intermediate_ll = []

    def log_likelihood(self, cov: float, nrows: int, ncols: int) -> float:
        """
        Compute the GGS log likelihood.

        Args
        ----
            cov: covariance
            nrows: number of observations
            ncols: number of columns

        Returns
        -------
            log_likelihood
        """
        (_, logdet) = np.linalg.slogdet(
            cov + float(self.lamb) * np.identity(ncols) / nrows
        )

        return nrows * logdet - float(self.lamb) * np.trace(
            np.linalg.inv(cov + float(self.lamb) * np.identity(ncols) / nrows)
        )

    def cumulative_log_likelihood(
        self, data: np.ndarray, change_points: List[int]
    ) -> float:
        """
        Calculate cumulative GGS log likelihood for all segments.

        Args
        ----
            data: time series data
            change_points: list of indexes of change points

        Returns
        -------
            log_likelihood: cumulative log likelihood
        """
        log_likelihood = 0
        for start, stop in zip(change_points[:-1], change_points[1:]):
            segment = data[start:stop, :]
            nrows, ncols = segment.shape
            cov = np.cov(segment.T, bias=True)
            log_likelihood -= self.log_likelihood(cov, nrows, ncols)
        return log_likelihood

    def add_new_change_point(self, data: np.ndarray) -> Tuple[int, float]:
        """
        Add break point.

        Returns
        -------
            index: change point index
            gained log likehood
        """
        # Initialize parameters
        m, n = data.shape
        orig_mean = np.mean(data, axis=0)
        orig_cov = np.cov(data.T, bias=True)
        origLL = self.log_likelihood(orig_cov, m, n)
        totSum = m * (orig_cov + np.outer(orig_mean, orig_mean))
        muLeft = data[0, :] / n
        muRight = (m * orig_mean - data[0, :]) / (m - 1)
        runSum = np.outer(data[0, :], data[0, :])
        # Loop through all samples
        # find point where breaking the segment would have the largest LL increase
        minLL = origLL
        minInd = 0
        for i in range(2, m - 1):
            # Update parameters
            runSum = runSum + np.outer(data[i - 1, :], data[i - 1, :])
            muLeft = ((i - 1) * muLeft + data[i - 1, :]) / (i)
            muRight = ((m - i + 1) * muRight - data[i - 1, :]) / (m - i)
            sigLeft = runSum / (i) - np.outer(muLeft, muLeft)
            sigRight = (totSum - runSum) / (m - i) - np.outer(muRight, muRight)

            # Compute Cholesky, LogDet, and Trace
            Lleft = np.linalg.cholesky(sigLeft + float(self.lamb) * np.identity(n) / i)
            Lright = np.linalg.cholesky(
                sigRight + float(self.lamb) * np.identity(n) / (m - i)
            )
            llLeft = 2 * sum(map(math.log, np.diag(Lleft)))
            llRight = 2 * sum(map(math.log, np.diag(Lright)))
            (trLeft, trRight) = (0, 0)
            if self.lamb > 0:
                trLeft = math.pow(np.linalg.norm(np.linalg.inv(Lleft)), 2)
                trRight = math.pow(np.linalg.norm(np.linalg.inv(Lright)), 2)
            LL = (
                i * llLeft
                - float(self.lamb) * trLeft
                + (m - i) * llRight
                - float(self.lamb) * trRight
            )
            # Keep track of the best point so far
            if LL < minLL:
                minLL = LL
                minInd = i
        # Return break, increase in LL
        return minInd, minLL - origLL

    def adjust_change_points(
        self, data: np.ndarray, change_points: List[int], newInd
    ) -> List[int]:
        """
        Adjust change points.

        Args
        ----
            data: time series data
            change_points: change points indexes
            newInd:

        Returns
        -------
            change_points:
        """
        bp = change_points[:]
        random.seed(0)
        # Just one breakpoint, no need to adjust anything
        if len(bp) == 3:
            return bp
        # Keep track of what change_points have changed,
        # so that we don't have to adjust ones which we know are constant
        lastPass = {}
        thisPass = {b: 0 for b in bp}
        for i in newInd:
            thisPass[i] = 1
        for _ in range(self.max_shuffles):
            lastPass = dict(thisPass)
            thisPass = {b: 0 for b in bp}
            switchAny = False
            ordering = list(range(1, len(bp) - 1))
            random.shuffle(ordering)
            for i in ordering:
                # Check if we need to adjust it
                if (
                    lastPass[bp[i - 1]] == 1
                    or lastPass[bp[i + 1]] == 1
                    or thisPass[bp[i - 1]] == 1
                    or thisPass[bp[i + 1]] == 1
                ):
                    tempData = data[bp[i - 1] : bp[i + 1], :]
                    ind, val = self.add_new_change_point(tempData)
                    if bp[i] != ind + bp[i - 1] and val != 0:
                        lastPass[ind + bp[i - 1]] = lastPass[bp[i]]
                        del lastPass[bp[i]]
                        del thisPass[bp[i]]
                        thisPass[ind + bp[i - 1]] = 1
                        if self.verbose:
                            logger.info(
                                f"Moving {bp[i]} to {ind + bp[i - 1]}"
                                f"length = {tempData.shape[0]}, {ind}"
                            )
                        bp[i] = ind + bp[i - 1]
                        switchAny = True
            if not switchAny:
                return bp
        return bp

    def initialize_change_points(self, data: np.ndarray) -> List[int]:
        """Initialize change points."""
        return [0, data.shape[0] + 1]

    def find_change_points(self, data: np.ndarray) -> List[int]:
        """
        Find ``k`` change points on the data at a specific lambda.

        Args
        ----
            data: time series data

        Returns
        -------
            The K change points, along with all intermediate change points (for k < K)
            and their corresponding covariance-regularized maximum likelihoods.
        """
        change_points = self.initialize_change_points(data)
        self.intermediate_change_points = [change_points[:]]
        self.intermediate_ll = [self.cumulative_log_likelihood(data, change_points)]

        # Start GGS Algorithm
        for _ in range(self.k_max):
            new_index = -1
            new_value = +1
            # For each segment, find change point and increase in LL
            for start, stop in zip(change_points[:-1], change_points[1:]):
                segment = data[start:stop, :]
                ind, val = self.add_new_change_point(segment)
                if val < new_value:
                    new_index = ind + start
                    new_value = val

            # Check if our algorithm is finished
            if new_value == 0:
                logger.info("Adding change points!")
                return change_points

            # Add new change point
            change_points.append(new_index)
            change_points.sort()
            if self.verbose:
                logger.info(f"Change point occurs at: {new_index}, LL: {new_value}")

            # Adjust current locations of the change points
            change_points = self.adjust_change_points(data, change_points, [new_index])[
                :
            ]

            # Calculate likelihood
            ll = self.cumulative_log_likelihood(data, change_points)
            self.intermediate_change_points.append(change_points[:])
            self.intermediate_ll.append(ll)

        return change_points

    def fit(self, X):
        change_points = self.find_change_points(X)
        labels = np.zeros(X.shape[0], dtype=np.int32)

        for i, (start, stop) in enumerate(zip(change_points[:-1], change_points[1:])):
            labels[start:stop] = i
        return labels


class SklearnBaseEstimator(metaclass=abc.ABCMeta):
    """Define the domain-specific interface that Client uses."""

    @abc.abstractmethod
    def fit(self):
        """Fit."""
        pass

    @abc.abstractmethod
    def predict(self):
        """Predict."""
        pass


# class GGSEstimator(SklearnBaseEstimator):
#     """Sklearn Adapter."""

#     adaptee_class = GGS
    
#     def __init__(
#         self, k_max: int, lamb: float, max_shuffles: int = 250, verbose: bool = False
#     ):
#         self.k_max = k_max
#         self.lamb = lamb
#         self.max_shuffles = max_shuffles
#         self.verbose = verbose

#     def fit(self, X, y=None):
#         """Fit."""
#         self.adaptee = self.adaptee_class(**self.kwargs)
#         self.adaptee.initialize_intermediates()
#         return self.adaptee

#     def predict(self, X, y=None):
#         """Predict."""
#         return self.fit(X, y).find_change_points(X)


class GGSEstimator(SklearnBaseEstimator):
    """Sklearn Adapter."""

    def __init__(self, **kwargs):
        self.adaptee_class = GGS
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Fit."""
        self.adaptee = self.adaptee_class(**self.kwargs)
        self.adaptee.initialize_intermediates()
        return self

    def predict(self, X, y=None):
        """Predict."""
        return self.fit(X, y).fit(X)

    def get_params(self):
        return self.kwargs

    def set_params(self, *args, *kwargs):
        self.adaptee_class(*args, **kwargs)
