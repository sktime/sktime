"""Package - pykalman.

=============
Kalman Module
=============

This module provides inference methods for state-space estimation in continuous
spaces.
"""

__authors__ = [
    "duckworthd",  # main author original pykalman package
    "mbalatsko",  # update ot python 3.11 and later, temporary maintainer
    "gliptak",  # minor updates
    "nils-werner",  # minor updates
    "jonathanng",  # minor docs fix
    "pierre-haessig",  # minor docs fix
]

from .standard import KalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter

__all__ = [
    "KalmanFilter",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "datasets",
    "sqrt",
]
