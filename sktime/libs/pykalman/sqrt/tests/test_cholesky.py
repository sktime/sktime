from unittest import TestCase

from pykalman.datasets import load_robot
from pykalman.sqrt import CholeskyKalmanFilter
from pykalman.tests.test_standard import KalmanFilterTests


class CholeskyKalmanFilterTestSuite(TestCase, KalmanFilterTests):
    """Run tests for the Cholesky Factorization-based Kalman Filter."""

    def setUp(self):
        self.KF = CholeskyKalmanFilter
        self.data = load_robot()
