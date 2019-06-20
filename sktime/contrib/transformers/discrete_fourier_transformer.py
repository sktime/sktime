# dummy transform
import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer
from enum import Enum



class DiscreteFourierType(Enum):
    STANDARD = 1
    REAL = 2
    HERMITIAN = 3


class DiscreteFourierTransformer(BaseTransformer):

    def __init__(self, fourier_type=DiscreteFourierType.STANDARD, axis=None, norm=None, check_input=True):
        if not isinstance(self.type, DiscreteFourierType):
            raise TypeError("type should be defined as a DiscreteFourierTransform type")

        self.check_input = check_input
        self.type = fourier_type
        self.norm = norm
        self.axis = axis

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")

        if self.type == 1:
            return np.fft.fftn(X, axis=self.axis, norm=self.norm)
        elif self.type == 2:
            return np.fft.rfftn(X, axis=self.axis, norm=self.norm)
        elif self.type == 3:
            return np.fft.hfft(X, axis=self.axis, norm=self.norm)
        pass