from unittest import TestCase

from pykalman.sqrt import CholeskyKalmanFilter
from pykalman.tests.test_standard import KalmanFilterTests
from pykalman.datasets import load_robot

class CholeskyKalmanFilterTestSuite(TestCase, KalmanFilterTests):
    """Run Kalman Filter tests on the Cholesky Factorization-based Kalman
    Filter"""

    def setUp(self):
        self.KF = CholeskyKalmanFilter
        self.data = load_robot()
