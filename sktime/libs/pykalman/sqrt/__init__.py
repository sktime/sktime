"""Kalman module for state-space estimation in continuous spaces.

=============
Kalman Module
=============

This module provides inference methods for state-space estimation in continuous
spaces.
"""

from .bierman import BiermanKalmanFilter
from .cholesky import CholeskyKalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter

__all__ = [
    "BiermanKalmanFilter",
    "CholeskyKalmanFilter",
    "AdditiveUnscentedKalmanFilter",
]
