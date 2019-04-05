import numpy as np
import pandas as pd
import Transformer
from sklearn.base import BaseEstimator


class PowerSpectrumTransformer(Transformer):

    def __init__(self, maxLag=100):
        self._maxLag = maxLag

    def transform(self, X):
        transformedX = np.copy(X)
        return transformedX

