#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np

from sktime.transformers.base import _SeriesToPrimitivesTransformer
from sktime.utils.validation.series import check_series


class MeanTransformer(_SeriesToPrimitivesTransformer):
    def transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.mean(Z, axis=0)
