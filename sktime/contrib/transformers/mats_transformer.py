import numpy as np
from sktime.transformers.base import BaseTransformer
from sktime.contrib.transformers.mapped_transformer_base import BaseMappedTransformer

__author__ = ["Jeremy Sellier"]


class DiscreteFourierTransformer(BaseMappedTransformer):

    def __init__(self, fourier_type='discreteFT', axis=None, norm=None, check_input=True):
        if fourier_type not in ['discreteFT', 'discreteRealFT', 'discreteHermitFT']:
            raise TypeError("unrecognized discrete fourier type")

        self.transform_parameters = {'axis': axis, 'norm': norm}

        self.type_ = fourier_type
        self.check_input_ = check_input
        self.is_fitted_ = True

    def __get_transform_params(self):
        return self.transform_parameters


class AutoCorrelationFunctionTransformer(BaseMappedTransformer):

    def __init__(self, unbiased=False, acf_type='standard', nlags=40, method='ywunbiased', qstat=False, fft=False,
                 alpha=None, missing='none', check_input=True):
        if acf_type == 'standard':
            self.type_ = 'stdACF'
            self.transform_parameters = {'unbiased': unbiased, 'nlags': nlags, 'qstat': qstat, 'fft': fft,
                                         'alpha': alpha, 'missing': missing}

        elif acf_type == 'partial':
            self.type_ = 'pACF'
            self.transform_parameters = {'unbiased': unbiased, 'nlags': nlags, 'method': method, 'alpha': alpha}
        else:
            raise TypeError("unrecognized ACF type")

        self.check_input_ = check_input
        self.is_fitted_ = True

    def __get_transform_params(self):
        return self.transform_parameters


class PowerSpectrumTransformer(BaseMappedTransformer):

    def __init__(self, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density',
                 axis=-1, check_input=True):

        self.transform_parameters = {'fs': fs, 'window': window, 'nfft': nfft, 'detrend': detrend,
                                     'return_onesided': return_onesided, 'scaling': scaling, 'axis': axis}

        self.type_ = 'powerSpectrum'
        self.check_input_ = check_input
        self.is_fitted_ = True

    def __get_transform_params(self):
        return self.transform_parameters


class CosineTransformer(BaseTransformer):

    def __init__(self):
        self.type = 'cosin'

    def transform(self, x, y=None):
        return np.cos(x)