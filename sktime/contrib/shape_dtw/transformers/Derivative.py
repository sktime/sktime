import numpy as np
import pandas as pd

import math

from sktime.transformers.base import BaseTransformer

class Derivative(BaseTransformer):

    def __init__(self):
        pass
        
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
        
        #Calculate the derivative of each time series.
        res = []
        for x in range(num_insts):
            temp = [X[x][i]-X[x][(i+1)] for i in range(num_atts-1)]
            res.append(temp)
        
        X = np.asarray(res)
        
        return X