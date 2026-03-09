from sktime.forecasting.base import BaseForecaster
import numpy as np


class MoiraiMOEForecaster(BaseForecaster):

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.last_value_ = y.iloc[-1]
        return self

    def _predict(self, fh, X=None):
        n = len(fh)
        return np.repeat(self.last_value_, n)