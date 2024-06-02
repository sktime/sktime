from unittest import TestCase

from ...datasets import load_robot
from ..cholesky import CholeskyKalmanFilter
from ...tests.test_standard import KalmanFilterTests


class CholeskyKalmanFilterTestSuite(TestCase, KalmanFilterTests):
    """Run tests for the Cholesky Factorization-based Kalman Filter."""

    def setUp(self):
        self.KF = CholeskyKalmanFilter
        self.data = load_robot()
