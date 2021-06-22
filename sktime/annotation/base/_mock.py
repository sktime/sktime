# -*- coding: utf-8 -*-
from sktime.annotation.base._base import BaseAnnotator


class MockAnnotator(BaseAnnotator):

    def __init__(self, n=1):

        self.n = n

        super().__init__()

    def _fit(self, X, Y=None, Z=None):

        return self

    def _predict(self, X, Y=None, Z=None):

        n = self.n

        Xt = X.copy()

        for Xi in Xt:
            Xi.iloc[:] = False
            if len(Xi) > n:
                Xt.iloc[n] = True

        return Xt
