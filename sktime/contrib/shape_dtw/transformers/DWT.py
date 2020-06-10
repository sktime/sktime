import numpy as np
import pandas as pd
import pywt
import math
from sktime.transformers.base import BaseTransformer
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts

"""
Module to calculate the Discrete Wavelet Transform of a time series.
This class is simply a wrapper for the pywt.wavedec() function.
"""
class DWT(BaseTransformer):

    def __init__(self,num_levels=3):
        self.num_levels=num_levels
        
    """
    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_samples, num_atts]
        The training input samples.  If a Pandas data frame is passed, the column 0 is extracted

    Returns
    -------
    dims: numpy array or sparse matrix of shape = [n_samples, num_atts]
    """
    def transform(self,X):
        #Check input
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("ShapeDTW cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0,0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects")
                
        #get the number of attributes and instances
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        #Extract the wavelet coefficients
        coeffs = pywt.wavedec(X,'haar',level=self.num_levels)
        #Define the number of coefficient arrays
        numCoeffArrs = self.num_levels+1

        #Concatenate all the coefficients together.
        transformedData = []
        for x in range(num_insts):
            temp = []
            for y in range(numCoeffArrs):
                temp.extend(coeffs[y][x])
            transformedData.append(temp)
        
        transformedData = np.asarray(transformedData)

        return transformedData
        

        
if __name__ == "__main__":
    trainPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_ts\\Chinatown\\Chinatown_TRAIN.ts"
    trainData,trainDataClasses =  load_ts(trainPath)
    
    dwt=DWT(num_levels=2)
    print(dwt.transform(trainData))
    