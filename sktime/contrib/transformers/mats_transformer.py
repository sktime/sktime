from sktime.contrib.transformers.mapped_transformer_base import BaseMappedTransformer

__author__ = ["Jeremy Sellier"]


class DiscreteFourierTransformer(BaseMappedTransformer):

    def __init__(self, fourier_type='discreteFT', axis=None, norm=None, check_input=True):
        self.check_valid_key(fourier_type)
        self.type = fourier_type
        self.transform_parameters = {'axis': axis, 'norm': norm}
        self.check_input = check_input
        self.is_fitted_ = True

    def get_transform_params(self):
        return self.transform_parameters


class AutoCorrelationFunctionTransformer(BaseMappedTransformer):

    def __init__(self, unbiased=False, nlags=40, qstat=False, fft=False, alpha=None, missing='none', check_input=True):
        self.type = 'ACF'
        self.transform_parameters = {'unbiased': unbiased, 'nlags': nlags, 'qstat': qstat, 'fft': fft, 'alpha': alpha,
                                     'missing': missing}
        self.check_input = check_input
        self.is_fitted_ = True

    def get_transform_params(self):
        return self.transform_parameters


class PowerSpectrumTransformer(BaseMappedTransformer):

    def __init__(self, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density',
                 axis=-1, check_input=True):
        self.type = 'powerSpectrum'
        self.transform_parameters = {'fs': fs, 'window': window, 'nfft': nfft, 'detrend': detrend,
                                     'return_onesided': return_onesided, 'scaling': scaling, 'axis': axis}
        self.check_input = check_input
        self.is_fitted_ = True

    def get_transform_params(self):
        return self.transform_parameters
