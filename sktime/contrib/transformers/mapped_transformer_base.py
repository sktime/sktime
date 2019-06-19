import numpy as np
import pandas as pd

from sktime.transformers.base import BaseTransformer
from sktime.contrib.transformers.mapped_function_config import FunctionConfigs

__author__ = ["Jeremy Sellier"]


""" Prototype mechanism for the 'mapped' transformer classes
(comment to be added)

-- works as a base abstract class
-- store a static ref to the function config class
-- contain a generic 'transform' method
-- just need to override a 'get_transform_params' on the sub-classes (i.e. that return a dict of parameters to be passed within the corresponding lambda-f)

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