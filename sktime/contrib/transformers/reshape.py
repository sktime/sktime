import numpy as np
import pandas as pd
import sys as sys
from sktime.transformers.base import BaseTransformer
from enum import Enum
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts
import os


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
        Note that the start and end point are set in fit (i.e. from the train data). This
        could mean that a Padder actually truncates in the test, or that we need to pad in a Truncator.
        It seems the only unbiased way to deal with this.
    """

    def __init__(self,type=ResizeType.PADDER, start=None, end=None, pad_value=0, min=0,max=None):
        self.type=type
        self.start=start
        self.end=end
        self.pad_value=0
        self.min=min
        self.max=max

    def fit(self,X,y=None):
        """
        :param X:
        :param y:
        :return:
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")
        # NOTE, this should be moved to the new TS_Pandas, when it is implemented
        self.min, self.max = min_max_lengths(X)

        if type == ResizeType.PADDER:
            self.start = 0
            self.end = self.max  # find_max_length(series_lengths)
        elif type == ResizeType.TRUNCATOR:
            self.start = 0
            self.end = self.min  # find_min_length(series_lengths)
        elif type == ResizeType.RESIZER:
            if self.start == None or self.end == None:
                raise TypeError("start and end have not been set for the resizer and must be integers")

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

#Create the new, empty panda
        Xt = pd.DataFrame()
#Can we do this without nested loops? Who knows!
#Copy over new values, dont think there is a quick way to do this.
        for i in range(0, num_dimensions):
            X = np.asarray([a.values for a in X.iloc[:,i]])
            Xnew=np.zeros((num_cases, self.end-self.start+1), dtype=int)
            Xnew[:,0:self.end-self.start+1]=X[:,0:self.end+1]
        Xt = pd.DataFrame(Xnew)


#Return new pandas
        return Xt


def min_max_lengths(X, dim_to_use=None):
    """
    Function to find the min and max length of all the series by default over all dimensions
    """
    num_cases, num_dimensions = X.shape
    min = sys.maxsize
    max = 0
    if dim_to_use is None: #Do all
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                l=len(X.iloc[i,j])
                if l > max:
                    max=l
                elif l< min:
                    min=l
    return min,max

if __name__ == "__main__":
    path = 'C:/Users/ajb/Dropbox/Code2019/Transformers/sktime/sktime/datasets/data/'
    dataset = "PLAID"
    fname = path+dataset + '/'+dataset+'_TRAIN.ts'
    trainX,trainY = load_ts(fname)
    print(trainX.shape)
    print(trainX.shape[1])
    pad = Resizer(type=ResizeType.PADDER)
    pad.fit(trainX)
    print("Min length " + str(pad.min) + " Max length " + str(pad.max))
    padTrainX=pad.transform(trainX)

    #
    fname = path+dataset + '/'+dataset+'_TEST.ts'
    testX,testY = load_ts(fname)


