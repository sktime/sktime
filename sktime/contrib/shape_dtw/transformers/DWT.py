import numpy as np
import pandas as pd
import pywt
import math
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts


__author__ = "Vincent Nicholson"

"""
Module to calculate the Discrete Wavelet Transform of a time series.
This class is simply a wrapper for the pywt.wavedec() function.
"""
class DWT(BaseSeriesAsFeaturesTransformer):

    def __init__(self,num_levels=3):
        self.num_levels=num_levels
        super(DWT, self).__init__()
        
    """
    Parameters
    ----------
    X : a pandas dataframe of shape = [n_samples, num_dims]
        The training input samples. 

    Returns
    -------
    dims: a pandas data frame of shape = [n_samples, num_dims]
    """
    def transform(self,X):
    
        #Check the data
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False)
        
        #Get information about the dataframe
        num_dims = X.shape[1]
        num_atts = len(X.iloc[0,0])
        num_insts = X.shape[0]
        col_names = X.columns
        
        dataFrames = []
        df = pd.DataFrame()
        
        for x in col_names:
            #Convert one of the columns in the dataframe to numpy array
            arr = tabularize(pd.DataFrame(X[x]), return_array=True)
        
            #Extract the wavelet coefficients
            coeffs = pywt.wavedec(arr,'haar',level=self.num_levels)
            #Define the number of coefficient arrays
            numCoeffArrs = self.num_levels+1

            #Concatenate all the coefficients together.
            transformedData = []
            for y in range(num_insts):
                temp = []
                for z in range(numCoeffArrs):
                    temp.extend(coeffs[z][y])
                transformedData.append(temp)
        
            #Convert to a numpy array
            transformedData = np.asarray(transformedData)
            
            #Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))
            
            df[x] = colToAdd

        return df
        
if __name__ == "__main__":
    #Test that this transformer works on multivariate data, not just within ShapeDTW.
    trainPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Multivariate2018_ts\\AtrialFibrillation\\AtrialFibrillation_TRAIN.ts"
    trainData,trainDataClasses =  load_ts(trainPath)
    
    d = DWT()
    d.fit(trainData)
    res = d.transform(trainData)
    
    print(trainData.iloc[1,1])
    print(res.iloc[1,1])
    