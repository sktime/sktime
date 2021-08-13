# -*- coding: utf-8 -*-
from sktime.dists_kernels._base import BasePairwiseTransformer


class SktimeDist(BasePairwiseTransformer):
    def _transform(self, X, X2=None):
        pass
