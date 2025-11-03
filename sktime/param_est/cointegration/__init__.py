"""Module for parameter estimators of cointegration tests.

This module provides functionality for selecting the coint_rank in VECM
(Vector Error Correction Models).

The main class ARLagOrderSelector implements lag order selection
using various information criteria (AIC, BIC, HQIC) and supports both
sequential and global search strategies.
"""

__all__ = [
    "JohansenCointegration",
]
from sktime.param_est.cointegration._johansen import JohansenCointegration
