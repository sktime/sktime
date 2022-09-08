# -*- coding: utf-8 -*-
"""
Greedy Gaussian Segmentation (GGS).

Based on the work from [1]_.

- source code adapted based on: https://github.com/cvxgrp/GGS
- paper available at: https://stanford.edu/~boyd/papers/pdf/ggs.pdf

References
----------
.. [1] Hallac, D., Nystrup, P. & Boyd, S.
   "Greedy Gaussian segmentation of multivariate time series.",
    Adv Data Anal Classif 13, 727–751 (2019).
    https://doi.org/10.1007/s11634-018-0335-0
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from attrs import asdict, define, field
from sklearn.utils.validation import check_random_state

from sktime.base import BaseEstimator

logger = logging.getLogger(__name__)


@define
class GGS:
    """
    Greedy Gaussian Segmentation.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find
    lamb: : float, default=1.0
        Regularization parameter lambda
    max_shuffles: int, default=250
        maximum number of shuffles
    verbose: bool, default=False
        If ``True`` verbose output is enabled.
    random_state: int or np.random.RandomState, default=None
        Either random seed or an instance of ``np.random.RandomState``

    Notes
    -----
    Based on the work from [1]_.

    - source code adapted based on: https://github.com/cvxgrp/GGS
    - paper available at: https://stanford.edu/~boyd/papers/pdf/ggs.pdf

    References
    ----------
    .. [1] Hallac, D., Nystrup, P. & Boyd, S.,
       "Greedy Gaussian segmentation of multivariate time series.",
       Adv Data Anal Classif 13, 727–751 (2019).
       https://doi.org/10.1007/s11634-018-0335-0
    """

    k_max: int = 10
    lamb: float = 1.0
    max_shuffles: int = 250
    verbose: bool = False
    random_state: int = None

    change_points_: npt.ArrayLike = field(init=False, default=[])
    _intermediate_change_points: List[List[int]] = field(init=False, default=[])
    _intermediate_ll: List[float] = field(init=False, default=[])

    def initialize_intermediates(self):
        """Fit."""
        self._intermediate_change_points = []
        self._intermediate_ll = []

    def log_likelihood(self, cov: float, nrows: int, ncols: int) -> float:
        """
        Compute the GGS log likelihood.

        Parameters
        ----------
        cov: float
            covariance
        nrows: int
            number of observations
        ncols: int
            number of columns

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
        self, data: npt.ArrayLike, change_points: List[int]
    ) -> float:
        """
        Calculate cumulative GGS log likelihood for all segments.

        Args
        ----
        data: array_like
            time series data
        change_points: list of ints
            list of indexes of change points

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

    def add_new_change_point(self, data: npt.ArrayLike) -> Tuple[int, float]:
        """
        Add change point.

        Parameters
        ----------
        data: array_like
            time series data

        Returns
        -------
        index: change point index
        gll: gained log likelihood
        """
        # Initialize parameters
        m, n = data.shape
        orig_mean = np.mean(data, axis=0)
        orig_cov = np.cov(data.T, bias=True)
        orig_ll = self.log_likelihood(orig_cov, m, n)
        totSum = m * (orig_cov + np.outer(orig_mean, orig_mean))
        mu_left = data[0, :] / n
        mu_right = (m * orig_mean - data[0, :]) / (m - 1)
        runSum = np.outer(data[0, :], data[0, :])
        # Loop through all samples
        # find point where breaking the segment would have the largest LL increase
        minLL = orig_ll
        new_index = 0
        for i in range(2, m - 1):
            # Update parameters
            runSum = runSum + np.outer(data[i - 1, :], data[i - 1, :])
            mu_left = ((i - 1) * mu_left + data[i - 1, :]) / (i)
            mu_right = ((m - i + 1) * mu_right - data[i - 1, :]) / (m - i)
            sigLeft = runSum / (i) - np.outer(mu_left, mu_left)
            sigRight = (totSum - runSum) / (m - i) - np.outer(mu_right, mu_right)

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
                new_index = i
        # Return break, increase in LL
        return new_index, minLL - orig_ll

    def adjust_change_points(
        self, data: npt.ArrayLike, change_points: List[int], new_index: List[int]
    ) -> List[int]:
        """
        Adjust change points.

        Parameters
        ----------
        data: array_like
            time series data
        change_points: list of ints
            change points indexes
        new_index: list of ints
            new change points

        Returns
        -------
        change_points: list of ints
            change point indexes
        """
        rng = check_random_state(self.random_state)
        bp = change_points[:]

        # Just one breakpoint, no need to adjust anything
        if len(bp) == 3:
            return bp
        # Keep track of what change_points have changed,
        # so that we don't have to adjust ones which we know are constant
        last_pass = {}
        this_pass = {b: 0 for b in bp}
        for i in new_index:
            this_pass[i] = 1
        for _ in range(self.max_shuffles):
            last_pass = dict(this_pass)
            this_pass = {b: 0 for b in bp}
            switch_any = False
            ordering = list(range(1, len(bp) - 1))
            rng.shuffle(ordering)
            for i in ordering:
                # Check if we need to adjust it
                if (
                    last_pass[bp[i - 1]] == 1
                    or last_pass[bp[i + 1]] == 1
                    or this_pass[bp[i - 1]] == 1
                    or this_pass[bp[i + 1]] == 1
                ):
                    tempData = data[bp[i - 1] : bp[i + 1], :]
                    ind, val = self.add_new_change_point(tempData)
                    if bp[i] != ind + bp[i - 1] and val != 0:
                        last_pass[ind + bp[i - 1]] = last_pass[bp[i]]
                        del last_pass[bp[i]]
                        del this_pass[bp[i]]
                        this_pass[ind + bp[i - 1]] = 1
                        if self.verbose:
                            logger.info(
                                f"Moving {bp[i]} to {ind + bp[i - 1]}"
                                f"length = {tempData.shape[0]}, {ind}"
                            )
                        bp[i] = ind + bp[i - 1]
                        switch_any = True
            if not switch_any:
                return bp
        return bp

    def initialize_change_points(self, data: npt.ArrayLike) -> List[int]:
        """Initialize change points."""
        return [0, data.shape[0] + 1]

    def find_change_points(self, data: npt.ArrayLike) -> List[int]:
        """
        Find ``k`` change points on the data at a specific lambda.

        Parameters
        ----------
        data: array_like
            time series data

        Returns
        -------
        The K change points, along with all intermediate change points (for k < K)
        and their corresponding covariance-regularized maximum likelihoods.
        """
        change_points = self.initialize_change_points(data)
        self._intermediate_change_points = [change_points[:]]
        self._intermediate_ll = [self.cumulative_log_likelihood(data, change_points)]

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
            self._intermediate_change_points.append(change_points[:])
            self._intermediate_ll.append(ll)

        return change_points

    def predict(self, X: npt.ArrayLike):
        """Predict."""
        self.change_points_ = self.find_change_points(X)

        labels = np.zeros(X.shape[0], dtype=np.int32)
        for i, (start, stop) in enumerate(
            zip(self.change_points_[:-1], self.change_points_[1:])
        ):
            labels[start:stop] = i
        return labels


class GreedyGaussianSegmentation(BaseEstimator):
    """Greedy Gaussian Segmentation Estimator.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find
    lamb: : float, default=1.0
        Regularization parameter lambda
    max_shuffles: int, default=250
        maximum number of shuffles
    verbose: bool, default=False
        If ``True`` verbose output is enabled.
    random_state: int or np.random.RandomState, default=None
        Either random seed or an instance of ``np.random.RandomState``

    Notes
    -----
    Based on the work from [1]_.

    - source code adapted based on: https://github.com/cvxgrp/GGS
    - paper available at: https://stanford.edu/~boyd/papers/pdf/ggs.pdf

    References
    ----------
    .. [1] Hallac, D., Nystrup, P. & Boyd, S.,
       "Greedy Gaussian segmentation of multivariate time series.",
       Adv Data Anal Classif 13, 727–751 (2019).
       https://doi.org/10.1007/s11634-018-0335-0
    """

    def __init__(
        self,
        k_max: int = 10,
        lamb: float = 1.0,
        max_shuffles: int = 250,
        verbose: bool = False,
        random_state: int = None,
    ):
        # this is ugly and necessary only because of dum `test_constructor`
        self.k_max = k_max
        self.lamb = lamb
        self.max_shuffles = max_shuffles
        self.verbose = verbose
        self.random_state = random_state

        self._adaptee_class = GGS
        self._adaptee = self._adaptee_class(
            k_max=k_max,
            lamb=lamb,
            max_shuffles=max_shuffles,
            verbose=verbose,
            random_state=random_state,
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None):
        """Fit method for compatibility with sklearn-type estimator interface.

        It sets the internal state of the estimator and returns the initialized
        instance.

        Parameters
        ----------
        X: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.
        y: array_like
            Placeholder for compatibility with sklearn-api, not used, default=None.
        """
        self._adaptee.initialize_intermediates()
        return self

    def predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.ArrayLike:
        """Perform segmentation.

        Parameters
        ----------
        X: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.
        y: array_like
            Placeholder for compatibility with sklearn-api, not used, default=None.

        Returns
        -------
        y_pred : array_like
            1D array with predicted segmentation of the same size as the first
            dimension of X. The numerical values represent distinct segments
            labels for each of the data points.
        """
        return self._adaptee.predict(X)

    def fit_predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.ArrayLike:
        """Perform fit and predict.

        Parameters
        ----------
        X: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.
        y: array_like
            Placeholder for compatibility with sklearn-api, not used, default=None.

        Returns
        -------
        y_pred : array_like
            1D array with predicted segmentation of the same size as the first
            dimension of X. The numerical values represent distinct segments
            labels for each of the data points.
        """
        return self.fit(X, y).predict(X, y)

    def get_params(self, deep: bool = True) -> Dict:
        """Return initialization parameters.

        Parameters
        ----------
        deep: bool
            Dummy argument for compatibility with sklearn-api, not used.

        Returns
        -------
        params: dict
            Dictionary with the estimator's initialization parameters, with
            keys being argument names and values being argument values.
        """
        return asdict(self._adaptee, filter=lambda attr, value: attr.init is True)

    def set_params(self, **parameters):
        """Set the parameters of this object.

        Parameters
        ----------
        parameters : dict
            Initialization parameters for th estimator.

        Returns
        -------
        self : reference to self (after parameters have been set)
        """
        for key, value in parameters.items():
            setattr(self._adaptee, key, value)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the estimator."""
        return self._adaptee.__repr__()
