from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import periodogram
from enum import Enum
import numpy as np
from sktime.transformers.base import BaseTransformer


__author__ = ["Jeremy Sellier"]

""" Config class for the mapped function
(comment to be added)

-- store the lambdas and some other infos in a static member dictionary
-- should work as a 'singleton' (unless I am missing something with Python - to be confirmed)
"""


class FunctionConfigs:

    class FuncType(Enum):
        DFT = 1
        DFT_REAL = 2,
        DFT_HERMITE = 3,
        POWER_SPECTRUM = 4,
        ACF = 5,
        PACF = 6

    lambdaContainer = {
        FuncType.DFT: lambda p: np.fft.fftn(**p),
        FuncType.DFT_REAL: lambda p: np.fft.rfftn(**p),
        FuncType.DFT_HERMITE: lambda p: np.fft.hfft(**p),
        FuncType.POWER_SPECTRUM: lambda p: periodogram(**p),
        FuncType.ACF: lambda p: acf(**p),
        FuncType.PACF: lambda p: pacf(**p)
    }

    constraintTo1D = {
        FuncType.POWER_SPECTRUM: True,
        FuncType.ACF: True,
        FuncType.PACF: True
    }

    indexReturn = {
        FuncType.POWER_SPECTRUM: 0
    }

    def __init__(self):
        pass


""" Prototype mechanism for the 'mapped' transformer classes
(comment to be added)

- works as a main base abstract class (even though I am not sure how to force an class to be abstract in Python)
- store a static ref to the function config class
- contain a generic 'transform' method
- just need to override a 'get_transform_params' on the sub-classes (i.e. that return a dict of parameters to be passed within the corresponding lambda-f)

    Class Attributes
    ----------

    configRef_ : FunctionConfigs object
        reference to the function configs

    Instance Attributes
    ----------

    type_ = enum of type FunctionConfigs.FuncType
        Specify the type of the function to be mapped
    input_key_ = string
        specify the 'X' string key to be used in the corresponding lambda (can be 'a', 'x')
    check_input_ = False
        specify whether some input_check method should be trigger
    """


class BaseMappedTransformer(BaseTransformer):
    configRef = FunctionConfigs()

    def __init__(self):
        self.type_ = None
        self.input_key_ = None
        self.check_input_ = False

    def get_transform_params(self):
        pass

    def transform(self, x, y=None):

        if self.check_input_:
            self.__check_input(x, y)

        if not self.is_fitted_:
            raise TypeError("transformer must be fitted before performing the transform")

        parameters = self.get_transform_params()

        if self.type_ not in self.configRef.constraintTo1D or self.configRef.constraintTo1D[self.type_] is False:
            parameters[self.input_key_] = x
            out = self.configRef.lambdaContainer[self.type_](parameters)
            out = self.__get_output_from(out)

            return out

        else:
            arr = []
            for index, row in x.iterrows():
                parameters[self.input_key_] = row
                out = self.configRef.lambdaContainer[self.type_](parameters)
                out = self.__get_output_from(np.array(out))
                arr.append(out)

            arr = np.asarray(arr)
            return arr

    def __get_output_from(self, x):
        if self.type_ not in self.configRef.indexReturn:
            return x
        else:
            index = self.configRef.indexReturn[self.type_]
            return x[index]

    @staticmethod
    def __check_input(x, y=None):
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")





""" Sub method for the mapped transform class
(comment to be added)

Sub concrete class method for the mapped transformer instantiation. 
Act as a simple input mapping/collector. I kept different sub-classes to enforce a control of the input per category
"""


class DiscreteFourierTransformer(BaseMappedTransformer):

    def __init__(self, fourier_type='standard', axes=None, norm=None, check_input=True):

        if fourier_type == 'standard':
            self.type_ = FunctionConfigs.FuncType.DFT
        elif fourier_type == 'real':
            self.type_ = FunctionConfigs.FuncType.DFT_REAL
        elif fourier_type == 'hermite':
            self.type_ = FunctionConfigs.FuncType.DFT_HERMITE
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
            self.type_ = FunctionConfigs.FuncType.ACF
            self.transform_parameters = {'unbiased': unbiased, 'nlags': nlags, 'qstat': qstat, 'fft': fft,
                                         'alpha': alpha, 'missing': missing}
        elif acf_type == 'partial':
            self.type_ = FunctionConfigs.FuncType.PACF
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

        self.type_ = FunctionConfigs.FuncType.POWER_SPECTRUM
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
