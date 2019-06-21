import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer
from enum import Enum


__all__ = ['Resizer']
__author__ = ["Tony Bagnall"]

class ResizeType(Enum):
    PADDER = 1
    TRUNCATOR = 2
    RESIZER = 3

class Resizer(BaseTransformer):
    """
        transformer to standardise the size of the data set in one of three ways.
        1. type == PADDER: Pads all series to the length of the longest series
        2. type == TRUNCATOR: Shortens every series to be the same size as the smallest
        3. type == RESIZER: Resizes so each series goes from start to end (inclusive). It will pad if necessary, and data at
        start will be at position 0 in the new series. If this type is set transform will throw an exception if start and end are not set.
            ----------
        in the case where there is an empty series, TRUNCATOR will make everything empty
    """

    def __init__(self,type=ResizeType.PADDER, start=None, end=None, pad_value=0):
        self.type=type
        self.start=start
        self.end=end
        self.pad_value=0
    # No need for fit for Padding transformer

    def setResizeType(self,type):
        if type=="PADDING":
            self.type = ResizeType.PADDER
        elif type=="TRUNCATOR":
            self.type = ResizeType.TRUNCATOR
        elif type=="RESIZER":
            self.type = ResizeType.RESIZER

    def setResizer(self,start,end):
        self.type = ResizeType.RESIZER
        self.start=start
        self.end=end

    def transform(self, X, y=None):
        """
        Transform X,

        Parameters
        ----------
        X : nested pandas DataFrame, multi dimensional, with series of potentially different lengths
        Returns
        -------
        Xt : pandas DataFrame
          A new pandas DataFrame with same number of rows (cases) and columns (dimensions) as
          the original. The resulting will have series of all the same length
        """
        # If we do, how is that defined? Cannot
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")
        num_cases, num_dimensions = X.shape
#NOTE, this should be moved to the new TS_Pandas, when it is implemented
        min,max=min_max_lengths(X)
        if type==ResizeType.PADDER:
            self.start=0
            self.end=max #find_max_length(series_lengths)
        elif type==ResizeType.TRUNCATOR:
            self.start=0
            self.end=min #find_min_length(series_lengths)
        elif type==ResizeType.RESIZER:
            if self.start==None or self.end == None:
                raise TypeError("start and end have not been set for the resizer and must be integers")
#Can we do this without nested loops? Who knows!

#Create the new panda
        Xt = pd.DataFrame(num_cases, num_dimensions)
#Copy over new values, dont think there is a quick way to do this.
        for i in range(0, num_dimensions):
            X = np.asarray([a.values for a in X.iloc[:,i]])
            Xnew=np.zeros((num_cases, self.end-self.start+1), dtype=int)
            Xnew[:,0:self.end-self.start+1]=X[:,0:self.end+1]
            for j in range(0,num_cases):
                Xt[i,j]=X[j]

#Return new pandas
        return Xt


def min_max_lengths(X, dim_to_use=None):
    """
    Function to find the min and max length of all the series by default over all dimensions
    """
    num_cases, num_dimensions = X.shape
    min = 100000 #sys.maxint
    max = 0
    if dim_to_use is None: #Do all
        for i in range(0, X.ndim):
            for j in range(0,X.shape):
                l=len(X[i,j])
                if l > max:
                    max=l
                elif l< min:
                    min=l
    return min,max

if __name__ == "__main__":
    dataset = "Gunpoint"
#    loc = 'C:\Users\Jeremy\PycharmProjects\transformers\sktime\datasets\data\GunPoint\GunPoint_TEST.ts'
#    train,test = load_ts(loc)
    print(dataset)
