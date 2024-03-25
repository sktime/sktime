from unittest import TestCase

from pykalman.sqrt import BiermanKalmanFilter
from pykalman.tests.test_standard import KalmanFilterTests
from pykalman.datasets import load_robot

class BiermanKalmanFilterTestSuite(TestCase, KalmanFilterTests):
    """Run Kalman Filter tests on the UDU' Decomposition-based Kalman Filter"""

    def setUp(self):
        self.KF = BiermanKalmanFilter
        self.data = load_robot()
