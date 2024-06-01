"""
=============
Kalman Module
=============

This module provides inference methods for state-space estimation in continuous
spaces.
"""

from .bierman import BiermanKalmanFilter
from .cholesky import CholeskyKalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter
