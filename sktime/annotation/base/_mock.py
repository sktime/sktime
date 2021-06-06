# -*- coding: utf-8 -*-
from sktime.annotation.base._base import BaseAnnotator


class MockAnnotator(BaseAnnotator):
    def __init__(self):
        pass

    def fit(self, Z, X=None):
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        Zt = Z.copy()
        Zt.iloc[:] = False
        Zt.iloc[1] = True
        return Zt
