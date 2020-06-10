import numpy as np
import pandas as pd
import math
import statistics
from sktime.transformers.base import BaseTransformer
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts


class Slope(BaseTransformer):

    def __init__(self,num_intervals=8):
        self.num_intervals=num_intervals
        
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
            res = self.getGradientsOfLines(X[x])
            temp.append(res)
            
        return np.asarray(temp)
        
    """
    Function to get the gradients of the line of best fits given a time series.
    
    Parameters
    ----------
    X : a numpy array of shape = [num_atts]
    
    Returns
    -------
    m : a numpy array of shape = [num_intervals]. It contains the gradients of the line of best fit
        for each interval in a time series.
    """
    def getGradientsOfLines(self,X):
        #Firstly, split the time series into approx equal length intervals
        splitTimeSeries = self.splitTimeSeries(X)
        gradients = []
        
        for x in range(len(splitTimeSeries)):
            gradients.append(self.getGradient(splitTimeSeries[x]))
            
        return gradients
        
        
    """
    Function to get the gradient of the line of best fit given a section of a time series.
    
    Equation adopted from: real-statistics.com/regression/total-least-squares
    Parameters
    ----------
    Y : a numpy array of shape = [interval_size]
    
    Returns
    -------
    m : an int corresponding to the gradient of the best fit line.
    """
    def getGradient(self,Y):
        #Create a list of content 1,2,3,4,...,len(Y) for the x coordinates.
        X = [(i+1) for i in range(len(Y))]
        
        #Calculate the mean of both lists
        meanX = statistics.mean(X)
        meanY = statistics.mean(Y)
        
        #Calculate the list (yi-mean(y))^2
        yminYbar = [(y-meanY)**2 for y in Y]
        #Calculate the list (xi-mean(x))^2        
        xminXbar = [(x-meanX)**2 for x in X] 
        
        #Sum them to produce w.
        w = sum(yminYbar) - sum(xminXbar)
        
        #Calculate the list (xi-mean(x))*(yi-mean(y))
        temp = []
        for x in range(len(X)):
            temp.append((X[x]-meanX)*(Y[x]-meanY))
            
        #Sum it and multiply by 2 to calculate r
        r = 2*sum(temp)
        
        if r==0:
            #remove nans
            m = 0
        else:
            #Gradient is defined as (w+sqrt(w^2+r^2))/r
            m = (w+math.sqrt(w**2+r**2))/r
        
        return m
        
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
    Function for checking the values of parameters inserted into Slope.
    
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

    
if __name__ == "__main__":
    trainPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_ts\\Chinatown\\Chinatown_TRAIN.ts"
    trainData,trainDataClasses =  load_ts(trainPath)
    
    slope=Slope()
    print(len(slope.transform(trainData)))