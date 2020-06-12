import numpy as np
import pandas as pd
import math
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X

__author__= "Vincent Nicholson"

class SubsequenceTransformer(BaseSeriesAsFeaturesTransformer):

    def __init__(self,subsequenceLength=5):
        self.subsequenceLength=subsequenceLength
        super(SubsequenceTransformer, self).__init__()
        
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
    
        #get the number of attributes and instances
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)
        
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        
        #Check the parameters are appropriate
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
            
        #Convert this into a panda's data frame
        df = pd.DataFrame()
        for i in range(len(subsequences)):
            inst = subsequences[i]
            data = []
            for j in range(len(inst)):
                data.append(pd.Series(inst[j]))
            df[i] = data
      
        return df.transpose()
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
                raise ValueError("subsequenceLength cannot be higher than the length of the time series.")
        else:
            raise ValueError("subsequenceLength must be an 'int'. Found '" + type(self.subsequenceLength).__name__ + "' instead.")

    
    