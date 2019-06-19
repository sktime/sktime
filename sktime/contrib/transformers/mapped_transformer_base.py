import numpy as np
import pandas as pd

from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from sktime.transformers.base import BaseTransformer

__author__ = ["Jeremy Sellier"]

""" Prototype mechanism for the 'mapped' transformer classes
-- works as a base abstract class and store a dict of lambda as static member
-- just need to override a 'get_transform_params on the sub-classes (that return a dict of parameters to be passed within the corresponding lambda-f)
"""


class BaseMappedTransformer(BaseTransformer):
    __mappingContainer = {
        'discreteFT': lambda p: np.fft.fftn(**p),
        'discreteRealFT': lambda p: np.fft.rfftn(**p),
        'discreteHermitFT': lambda p: np.fft.hfft(**p),
        'powerSpectrum': lambda p: periodogram(**p),
        'stdACF': lambda p: acf(**p),
        'pACF': lambda p: pacf(**p)
    }

    __constraint1D = {
        'powerSpectrum': True,
        'stdACF': True,
        'pACF': True
    }

    __indexReturn = {
        'powerSpectrum': 0
    }

    def __get_transform_params(self, x, y=None):
        pass

    def transform(self, x, y=None):
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")

        if not self.is_fitted_:
            raise TypeError("transformer must be fitted before performing the transform")

        parameters = self.__get_transform_params()

        if self.type not in self.__constraint1D or self.__constraint1D[self.type] is False:
            parameters['x'] = x
            print('parameters')
            out = self.__mappingContainer[self.type](parameters)
            out = self.__get_output_from(out)

            return out

        else:
            arr = []
            for index, row in x.iterrows():
                parameters['x'] = row
                out = self.__mappingContainer[self.type](parameters)
                out = self.__get_output_from(np.array(out))
                arr.append(out)

            arr = np.asarray(arr)
            return arr

    def __get_output_from(self, x):
        if self.type not in self.__indexReturn:
            return x
        else:
            index = self.__indexReturn[self.type]
            return x[index]

