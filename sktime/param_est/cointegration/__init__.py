"""Module for parameter estimators of cointegration tests.

This module provides functionality for selecting the coint_rank in VECM
(Vector Error Correction Models).
"""

__all__ = [
    "JohansenCointegration",
]
from sktime.param_est.cointegration._johansen import JohansenCointegration
