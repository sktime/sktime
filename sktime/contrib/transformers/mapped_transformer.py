import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer


class MappedTransformer(BaseTransformer):

    mappingContainer_ = {
        'discreteFT': lambda a: np.fft.fftn(a),
        'discreteRealFT': lambda a: np.fft.rfftn(a),
        'discreteHermitFT': lambda a: np.fft.hfft(a),
    }

    def check_valid_key(self, key_entry=''):
        if key_entry not in self.mappingContainer_:
            raise TypeError("type should be part of the predefined mapped name")
        pass

    def get_transform_params(self, X, y=None):
        pass

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")

        parameters = self.get_transform_params(X, y)
        return self.mappingContainer_[self.type](parameters)


class DiscreteFourierTransformer(MappedTransformer):

    def __init__(self, fourier_type='discreteFT', axis=None, norm=None, check_input=True):
        self.check_valid_key(fourier_type)
        self.check_input = check_input
        self.type = fourier_type
        self.norm = norm
        self.axis = axis

    def get_transform_params(self, X, y=None):
        return {'x': X, 'y': y, 'axis': self.axis, 'norm': self.norm}
