'''
=============
Kalman Module
=============

This module provides inference methods for state-space estimation in continuous
spaces.
'''

from .cholesky import CholeskyKalmanFilter
from .bierman import BiermanKalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter
