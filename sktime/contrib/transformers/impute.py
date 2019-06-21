import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer
from enum import Enum


__all__ = ['Imputer']
__author__ = ["Tony Bagnall"]

class ImputeType(Enum):
    LINEAR = 1

class Imputer(BaseTransformer):
    """
        transformer that estimates missing values from the data. It does not structurally modify the
        data in any way. Missing is defined as ....
        1. type == LINEAR: linearly fits between values either side of missing segment. If the missing segment
        is at the start or end of the series, it uses the series mean as the start/end point
            ----------
    """

    def __init__(self,type=ImputeType.LINEAR, in_place=True):
        self.type=type
        self.in_place=in_place

    def transform(self, X, y=None):
        """
        Transform X,

        Parameters
        ----------
        X : nested pandas DataFrame, multi dimensional, with series of potentially different lengths
        and missing values
        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same structure as the original, but with no missing valuesh
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")
        num_cases, num_dimensions = X.shape
# Clone the data or impute in place. We could impute in place, but it is dangerous
        if self.in_place:
            Xt=X
        else:
#Create a new panda
            Xt = X.copy(deep=True)
#iterate accross all series to find all missing intervals
        for i in range(0, num_dimensions):
            X = np.asarray([a.values for a in X.iloc[:,i]])
            Xnew=np.zeros((num_cases, self.end-self.start+1), dtype=int)
            Xnew[:,0:self.end-self.start+1]=X[:,0:self.end+1]
            for j in range(0,num_cases):
                Xt[i,j].fillna(inplace=True)

#Return new pandas
        return Xt

