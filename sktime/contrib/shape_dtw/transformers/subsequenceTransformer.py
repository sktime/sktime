import numpy as np
import pandas as pd

import math

from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer

class SubsequenceTransformer(BaseSeriesAsFeaturesTransformer):

    def __init__(self,subsequenceLength=5):
        self.subsequenceLength=subsequenceLength
        
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
        
        self.checkParameters(num_atts)
        
        pad_amnt = math.floor(self.subsequenceLength/2)
        padded_data = np.zeros((num_insts,num_atts + (2*pad_amnt)))

        #Pad both ends of X
        for i in range(num_insts):
            padded_data[i] = np.pad(X[i],pad_amnt,mode='edge')
            
        subsequences = np.zeros((num_insts,num_atts,self.subsequenceLength))
        
        #Extract subsequences
        for i in range(num_insts):
            subsequences[i] = self.extractSubsequences(padded_data[i],num_atts)
      
        return subsequences
    """
    Function to extract a set of subsequences from a list of instances.
    
    Adopted from - https://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-construction-from-1d-array/4924433#4924433
    
    """
    def extractSubsequences(self,instance,num_atts):
        shape = (num_atts,self.subsequenceLength)
        strides = (instance.itemsize,instance.itemsize)
        return np.lib.stride_tricks.as_strided(instance, shape=shape, strides=strides)
    
    """
    Function for checking the values of parameters inserted into HOG1D.
    
    Throws
    ------
    ValueError if a parameters input is invalid.
    """
    def checkParameters(self,num_atts):
        if isinstance(self.subsequenceLength,int):
            if self.subsequenceLength <=0:
                raise ValueError("subsequenceLength must have the value of at least 1")
            if self.subsequenceLength > num_atts:
                raise ValueError("subsequenceLength cannot be higher than num_atts")
        else:
            raise ValueError("subsequenceLength must be an 'int'. Found '" + type(self.subsequenceLength).__name__ + "' instead.")

    
    