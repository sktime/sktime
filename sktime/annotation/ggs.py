"""Greedy Gaussian Segmentation (GGS).

The method approximates solutions for the problem of breaking a
multivariate time series into segments, where the data in each segment
could be modeled as independent samples from a multivariate Gaussian
distribution. It uses a dynamic programming search algorithm with
a heuristic that allows finding approximate solution in linear time with
respect to the data length and always yields locally optimal choice.

This module is structured with the ``GGS`` that implements the actual
segmentation algorithm and a ``GreedyGaussianSegmentation`` that
interfaces the algorithm with the sklearn/sktime api. The benefit
behind that design is looser coupling between the logic and the
interface introduced to allow for easier changes of either part
since segmentation still has an experimental nature. When making
algorithm changes you probably want to look into ``GGS`` when
evolving the sktime/sklearn interface look into ``GreedyGaussianSegmentation``.
This design also allows adapting ``GGS`` to other interfaces.

Notes
-----
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
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_random_state

from sktime.base import BaseEstimator
from sktime.utils.validation._dependencies import _check_estimator_deps

logger = logging.getLogger(__name__)


@dataclass
class GGS:
    """Greedy Gaussian Segmentation.

    The method approxmates solutions for the problem of breaking a
    multivariate time series into segments, where the data in each segment
    could be modeled as independent samples from a multivariate Gaussian
    distribution. It uses a dynamic programming search algorithm with
    a heuristic that allows finding approximate solution in linear time with
    respect to the data length and always yields locally optimal choice.

    Greedy Gaussian Segmentation (GGS) fits a segmented gaussian model (SGM)
    to the data by computing the approximate solution to the combinatorial
    problem of finding the approximate covariance-regularized  maximum
    log-likelihood for fixed number of change points and a reagularization
    strength. It follows an interative procedure
    where a new breakpoint is added and then adjusting all breakpoints to
    (approximately) maximize the objective. It is similar to the top-down
    search used in other change point detection problems.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.
    lamb: : float, default=1.0
        Regularization parameter lambda (>= 0), which controls the amount of
        (inverse) covariance regularization, see Eq (1) in [1]_. Regularization
        is introduced to reduce issues for high-dimensional problems. Setting
        ``lamb`` to zero will ignore regularization, whereas large values of
        lambda will favour simpler models.
    max_shuffles: int, default=250
        Maximum number of shuffles
    verbose: bool, default=False
        If ``True`` verbose output is enabled.
    random_state: int or np.random.RandomState, default=None
        Either random seed or an instance of ``np.random.RandomState``

    Attributes
    ----------
    change_points_: array_like, default=[]
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.
    _intermediate_change_points: List[List[int]], default=[]
        Intermediate values of change points for each value of k = 1...k_max
    _intermediate_ll: List[float], default=[]
        Intermediate values for log-likelihood for each value of k = 1...k_max

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

    change_points_: npt.ArrayLike = field(init=False, default_factory=list)
    _intermediate_change_points: List[List[int]] = field(
        init=False, default_factory=list
    )
    _intermediate_ll: List[float] = field(init=False, default_factory=list)

    def initialize_intermediates(self) -> None:
        """Initialize the state fo the estimator."""
        self._intermediate_change_points = []
        self._intermediate_ll = []

    def log_likelihood(self, data: npt.ArrayLike) -> float:
        """Compute the GGS log-likelihood of the segmented Gaussian model.

        Parameters
        ----------
        data: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.

        Returns
        -------
        log_likelihood
        """
        nrows, ncols = data.shape
        cov = np.cov(data.T, bias=True)
        (_, logdet) = np.linalg.slogdet(
            cov + float(self.lamb) * np.identity(ncols) / nrows
        )

        return nrows * logdet - float(self.lamb) * np.trace(
            np.linalg.inv(cov + float(self.lamb) * np.identity(ncols) / nrows)
        )

    def cumulative_log_likelihood(
        self, data: npt.ArrayLike, change_points: List[int]
    ) -> float:
        """Calculate cumulative GGS log-likelihood for all segments.

        Args
        ----
        data: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.
        change_points: list of ints
            Locations of change points as integer indexes.
            By convention, change points
            include the identity segmentation, i.e. first and last index + 1 values.

        Returns
        -------
        log_likelihood: cumulative log likelihood
        """
        log_likelihood = 0
        for start, stop in zip(change_points[:-1], change_points[1:]):
            segment = data[start:stop, :]
            log_likelihood -= self.log_likelihood(segment)
        return log_likelihood

    def add_new_change_point(self, data: npt.ArrayLike) -> Tuple[int, float]:
        """Add change point.

        This methods finds a new change point by that splits the segment and
        optimizes the objective function. See section 3.1 on split subroutine
        in [1]_.

        Parameters
        ----------
        data: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.

        Returns
        -------
        index: change point index
        gll: gained log likelihood
        """
        # Initialize parameters
        m, n = data.shape
        orig_mean = np.mean(data, axis=0)
        orig_cov = np.cov(data.T, bias=True)
        orig_ll = self.log_likelihood(data)
        total_sum = m * (orig_cov + np.outer(orig_mean, orig_mean))
        mu_left = data[0, :] / n
        mu_right = (m * orig_mean - data[0, :]) / (m - 1)
        runSum = np.outer(data[0, :], data[0, :])
        # Loop through all samples
        # find point where breaking the segment would have the largest LL increase
        min_ll = orig_ll
        new_index = 0
        for i in range(2, m - 1):
            # Update parameters
            runSum = runSum + np.outer(data[i - 1, :], data[i - 1, :])
            mu_left = ((i - 1) * mu_left + data[i - 1, :]) / (i)
            mu_right = ((m - i + 1) * mu_right - data[i - 1, :]) / (m - i)
            sigLeft = runSum / (i) - np.outer(mu_left, mu_left)
            sigRight = (total_sum - runSum) / (m - i) - np.outer(mu_right, mu_right)

            # Compute Cholesky, LogDet, and Trace
            Lleft = np.linalg.cholesky(sigLeft + float(self.lamb) * np.identity(n) / i)
            Lright = np.linalg.cholesky(
                sigRight + float(self.lamb) * np.identity(n) / (m - i)
            )
            ll_left = 2 * sum(map(math.log, np.diag(Lleft)))
            ll_right = 2 * sum(map(math.log, np.diag(Lright)))
            (trace_left, trace_right) = (0, 0)
            if self.lamb > 0:
                trace_left = math.pow(np.linalg.norm(np.linalg.inv(Lleft)), 2)
                trace_right = math.pow(np.linalg.norm(np.linalg.inv(Lright)), 2)
            LL = (
                i * ll_left
                - float(self.lamb) * trace_left
                + (m - i) * ll_right
                - float(self.lamb) * trace_right
            )
            # Keep track of the best point so far
            if LL < min_ll:
                min_ll = LL
                new_index = i
        # Return break, increase in LL
        return new_index, min_ll - orig_ll

    def adjust_change_points(
        self, data: npt.ArrayLike, change_points: List[int], new_index: List[int]
    ) -> List[int]:
        """Adjust change points.

        This method adjusts the positions of all change points until the
        result is 1-OPT, i.e., no change of any one breakpoint improves
        the objective.

        Parameters
        ----------
        data: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.
        change_points: list of ints
            Locations of change points as integer indexes.
            By convention, change points
            include the identity segmentation, i.e. first and last index + 1 values.
        new_index: list of ints
            New change points

        Returns
        -------
        change_points: list of ints
            Locations of change points as integer indexes.
            By convention, change points
            include the identity segmentation, i.e. first and last index + 1 values.
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

    def identity_segmentation(self, data: npt.ArrayLike) -> List[int]:
        """Initialize change points."""
        return [0, data.shape[0] + 1]

    def find_change_points(self, data: npt.ArrayLike) -> List[int]:
        """
        Search iteratively  for up to ``k_max`` change points.

        Parameters
        ----------
        data: array_like
            2D `array_like` representing time series with sequence index along
            the first dimension and value series as columns.

        Returns
        -------
        The K change points, along with all intermediate change points (for k < K)
        and their corresponding covariance-regularized maximum likelihoods.
        """
        change_points = self.identity_segmentation(data)
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
            change_points = self.adjust_change_points(data, change_points, [new_index])
            change_points = change_points[:]

            # Calculate likelihood
            ll = self.cumulative_log_likelihood(data, change_points)
            self._intermediate_change_points.append(change_points[:])
            self._intermediate_ll.append(ll)

        return change_points


class GreedyGaussianSegmentation(BaseEstimator):
    """Greedy Gaussian Segmentation Estimator.

    The method approxmates solutions for the problem of breaking a
    multivariate time series into segments, where the data in each segment
    could be modeled as independent samples from a multivariate Gaussian
    distribution. It uses a dynamic programming search algorithm with
    a heuristic that allows finding approximate solution in linear time with
    respect to the data length and always yields locally optimal choice.

    Greedy Gaussian Segmentation (GGS) fits a segmented gaussian model (SGM)
    to the data by computing the approximate solution to the combinatorial
    problem of finding the approximate covariance-regularized  maximum
    log-likelihood for fixed number of change points and a reagularization
    strength. It follows an interative procedure
    where a new breakpoint is added and then adjusting all breakpoints to
    (approximately) maximize the objective. It is similar to the top-down
    search used in other change point detection problems.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.
    lamb: : float, default=1.0
        Regularization parameter lambda (>= 0), which controls the amount of
        (inverse) covariance regularization, see Eq (1) in [1]_. Regularization
        is introduced to reduce issues for high-dimensional problems. Setting
        ``lamb`` to zero will ignore regularization, whereas large values of
        lambda will favour simpler models.
    max_shuffles: int, default=250
        Maximum number of shuffles
    verbose: bool, default=False
        If ``True`` verbose output is enabled.
    random_state: int or np.random.RandomState, default=None
        Either random seed or an instance of ``np.random.RandomState``

    Attributes
    ----------
    change_points_: array_like, default=[]
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.
    _intermediate_change_points: List[List[int]], default=[]
        Intermediate values of change points for each value of k = 1...k_max
    _intermediate_ll: List[float], default=[]
        Intermediate values for log-likelihood for each value of k = 1...k_max

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

        _check_estimator_deps(self)
        super().__init__()

        self._adaptee = GGS(
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
        self.change_points_ = self._adaptee.find_change_points(X)

        labels = np.zeros(X.shape[0], dtype=np.int32)
        for i, (start, stop) in enumerate(
            zip(self.change_points_[:-1], self.change_points_[1:])
        ):
            labels[start:stop] = i
        return labels

    def fit_predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.ArrayLike:
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
        attrs_to_ignore = [
            "change_points_",
            "_intermediate_change_points",
            "_intermediate_ll",
        ]
        params = asdict(self._adaptee)
        params = {
            key: value for key, value in params.items() if key not in attrs_to_ignore
        }
        return params

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
