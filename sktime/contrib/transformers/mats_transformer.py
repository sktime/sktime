import numpy as np
from sktime.transformers.base import BaseTransformer
from sktime.contrib.transformers.mapped_transformer_base import BaseMappedTransformer
from sktime.contrib.transformers.mapped_function_config import FunctionConfigs


__author__ = ["Jeremy Sellier"]

""" Sub method for the mapped transform class
(comment to be added)

Sub concrete class method for the mapped transformer instantiation. 
Act as a simple input mapping/collector. I kept different sub-classes to enforce a control of the input per category
"""


class DiscreteFourierTransformer(BaseMappedTransformer):

    def __init__(self, fourier_type='standard', axes=None, norm=None, check_input=True):

        if fourier_type == 'standard':
            self.type_ = FunctionConfigs.FuncType.discreteFT
        elif fourier_type == 'real':
            self.type_ = FunctionConfigs.FuncType.discreteRealFT
        elif fourier_type == 'hermite':
            self.type_ = FunctionConfigs.FuncType.discreteHermiteFT
        else:
            raise TypeError("unrecognized discrete fourier type")

        self.transform_parameters = {'axes': axes, 'norm': norm}
        self.input_key_ = 'a'
        self.check_input_ = check_input
        self.is_fitted_ = True

    def get_transform_params(self):
        return self.transform_parameters


class AutoCorrelationFunctionTransformer(BaseMappedTransformer):

    def __init__(self, unbiased=False, acf_type='standard', nlags=40, method='ywunbiased', qstat=False, fft=False,
                 alpha=None, missing='none', check_input=True):

        if acf_type == 'standard':
            self.type_ = FunctionConfigs.FuncType.stdACF
            self.transform_parameters = {'unbiased': unbiased, 'nlags': nlags, 'qstat': qstat, 'fft': fft,
                                         'alpha': alpha, 'missing': missing}
        elif acf_type == 'partial':
            self.type_ = FunctionConfigs.FuncType.pACF
            self.transform_parameters = {'unbiased': unbiased, 'nlags': nlags, 'method': method, 'alpha': alpha}
        else:
            raise TypeError("unrecognized ACF type")

        self.input_key_ = 'x'
        self.check_input_ = check_input
        self.is_fitted_ = True

    def get_transform_params(self):
        return self.transform_parameters


class PowerSpectrumTransformer(BaseMappedTransformer):

    def __init__(self, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density',
                 axis=-1, check_input=True):
        self.transform_parameters = {'fs': fs, 'window': window, 'nfft': nfft, 'detrend': detrend,
                                     'return_onesided': return_onesided, 'scaling': scaling, 'axis': axis}

        self.type_ = FunctionConfigs.FuncType.powerSpectrum
        self.input_key_ = 'x'
        self.check_input_ = check_input
        self.is_fitted_ = True

    def get_transform_params(self):
        return self.transform_parameters


class CosineTransformer(BaseTransformer):

    def __init__(self):
        pass

    def transform(self, x, y=None):
        return np.cos(x)
