import numpy as np
import pandas as pd
import numbers
import math

from sktime.transformers.base import BaseTransformer

"""
The HOG1D Transformer proposed by:

@article{zhao2015classifying,
  title={Classifying time series using local descriptors with hybrid sampling},
  author={Zhao, Jiaping and Itti, Laurent},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={28},
  number={3},
  pages={623--637},
  year={2015},
  publisher={IEEE}
}
"""
class HOG1D(BaseTransformer):

    def __init__(self,num_intervals=2,num_bins=8,scaling_factor=0.1):
        self.num_intervals=num_intervals
        self.num_bins=num_bins
        self.scaling_factor=scaling_factor
        
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
        
        temp = []
        for x in range(num_insts):
            res = self.calculateHOG1Ds(X[x])
            temp.append(res)
            
            
        return np.asarray(temp)
        
    """
    Function to calculate the HOG1Ds given a time series.
    
    Parameters
    ----------
    X : a numpy array of shape = [num_atts]
    
    Returns
    -------
    m : a numpy array of shape = [num_intervals*num_bins]. It contains the histogram of each gradient within each interval.
    """
    def calculateHOG1Ds(self,X):
        #Firstly, split the time series into approx equal length intervals
        splitTimeSeries = self.splitTimeSeries(X)
        HOG1Ds = []
        
        for x in range(len(splitTimeSeries)):
            HOG1Ds.extend(self.getHOG1D(splitTimeSeries[x]))
            
        return HOG1Ds
        
    """
    Function to get the HOG1D given a portion of a time series.
    
    X : a numpy array of shape = [interval_size]
    
    Returns
    -------
    res : a numpy array of shape = [num_bins].
    """
    def getHOG1D(self,X):
        #First step is to pad the portion on both ends once.
        gradients=[0.0]*(len(X))
        X=np.pad(X,1,mode='edge')
        histogram=[0.0]*self.num_bins
        
        #Calculate the gradients of each element
        for i in range(1,len(X)-1):
            gradients[(i-1)] = self.scaling_factor*0.5*(X[(i+1)]-X[(i-1)])
        
        #Calculate the orientations
        orients = [math.degrees(math.atan(x)) for x in gradients]
        
        #Calculate the boundaries of the histogram
        hisBoundaries = [-90+(180/self.num_bins)+((180/self.num_bins)*x) for x in range(self.num_bins)]
        
        #Construct the histogram
        for x in range(len(orients)):
            orientToAdd = orients[x]
            for y in range(len(hisBoundaries)):
                if(orientToAdd<= hisBoundaries[y]):
                    histogram[y]+=1.0
                    break
        
        return histogram
        
    """
    Function to split a time series into approximately equal intervals.
    
    Adopted from = https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    
    Parameters
    ----------
    X : a numpy array corresponding to the time series being split into approx equal length intervals.
    """
    def splitTimeSeries(self,X):
        avg = len(X) / float(self.num_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning):int(beginning + avg)])
            beginning += avg

        return output
        
    """
    Function for checking the values of parameters inserted into HOG1D.
    
    Throws
    ------
    ValueError if a parameters input is invalid.
    """
    def checkParameters(self,num_atts):
        if isinstance(self.num_intervals,int):
            if self.num_intervals <=0:
                raise ValueError("num_intervals must have the value of at least 1")
            if self.num_intervals > num_atts:
                raise ValueError("num_intervals cannot be higher than num_atts")
        else:
            raise ValueError("num_intervals must be an 'int'. Found '" + type(self.num_intervals).__name__ + "' instead.")
            
        if isinstance(self.num_bins,int):
            if self.num_intervals <=0:
                raise ValueError("num_bins must have the value of at least 1")
        else:
            raise ValueError("num_bins must be an 'int'. Found '" + type(self.num_bins).__name__ + "' instead.")
            
        if not isinstance(self.scaling_factor,numbers.Number):
            raise ValueError("scaling_factor must be a 'number'. Found '" + type(self.scaling_factor).__name__ + "' instead.")
