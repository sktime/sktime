# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering._base import BaseClusterer, TimeSeriesInstances
from sktime.clustering.dnn.encoders._base import AutoEncoder


class IDEC(BaseClusterer):
    def __init__(self, auto_encoder: AutoEncoder):
        pass

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    def _fit(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass
