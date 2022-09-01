# -*- coding: utf-8 -*-
"""Preprocessing algorithm DOBIN (Distance based Outlier BasIs using Neighbors)."""

import numpy as np

__author__ = ["KatieBuc"]
__all__ = ["DOBIN"]

class DOBIN(BaseTransformer):
    """

    Parameters
    ----------


    Attributes
    ----------


    References
    ----------


    Examples
    --------
    """

    def __init__(
        self,
        alpha=0.01,
        k=10,
        knn_algorithm="brute",
        normalize=unitize,
        p=0.5,
        size_threshold=50,
        outlier_tail="max",
    ):
        self.alpha = alpha
        self.k = k
        self.knn_algorithm = knn_algorithm
        self.normalize = normalize
        self.p = p
        self.size_threshold = size_threshold
        self.outlier_tail = outlier_tail
        super(STRAY, self).__init__()

    def():
        