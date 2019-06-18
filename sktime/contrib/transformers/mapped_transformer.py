import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer


class MappedTransformer(BaseTransformer):

    mappingContainer_ = {
        'discreteFT': lambda X, axis, norm: np.fft.fftn(X, axis=axis, norm=norm),
        'discreteRealFT': lambda X, axis, norm: np.fft.rfftn(X, axis=axis, norm=norm),
        'discreteHermitFT': lambda X, axis, norm: np.fft.hfft(X, axis=axis, norm=norm),
    }
    pass


class DiscreteFourierTransformer(MappedTransformer):

    def __init__(self, fourier_type='discreteFT', axis=None, norm=None, check_input=True):
        self.check_input = check_input
        self.type = fourier_type
        self.norm = norm
        self.axis = axis

        if self.type not in self.mappingContainer_:
            raise TypeError("type should be part of the predefined mapped name")

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")

        return self.mappingContainer_[self.type](X, self.axis, self.norm)

