# -*- coding: utf-8 -*-
import numpy as np

# import pandas as pd

from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class CosineTransformer(_SeriesToSeriesTransformer):
    def transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.cos(Z)

    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.arccos(Z)
