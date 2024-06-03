from unittest import TestCase

from ...datasets import load_robot
from ...tests.test_standard import KalmanFilterTests
from ..bierman import BiermanKalmanFilter


class BiermanKalmanFilterTestSuite(TestCase, KalmanFilterTests):
    """Run Kalman Filter tests on the UDU' Decomposition-based Kalman Filter"""

    def setUp(self):
        self.KF = BiermanKalmanFilter
        self.data = load_robot()
