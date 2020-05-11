import numpy as np
import pandas as pd

from sktime.transformers.base import BaseTransformer
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.data_container import tabularize

__author__ = "Matthew Middlehurst"


class PAA(BaseTransformer):
    """ (PAA) Piecewise Aggregate Approximation Transformer, as described in
 Eamonn Keogh, Kaushik Chakrabarti, Michael Pazzani, and Sharad Mehrotra. 
 Dimensionality reduction for fast similarity search in large time series
 databases.
 Knowledge and information Systems, 3(3), 263-286, 2001.  
 For each series reduce the dimensionality to num_intervals, where each
 value is the mean of values in
 the interval. 
TO DO: pythonise it to make it more efficient. Maybe check vs this version
        http://vigne.sh/posts/piecewise-aggregate-approx/
Could have: Tune the interval size in fit somehow?
        
    Parameters
    ----------
    num_intervals   : int, dimension of the transformed data (default 8)

         """

    def __init__(self,
                 num_intervals=8
                 ):
        self.num_intervals = num_intervals
        super(PAA, self).__init__()

    def set_num_intervals(self, n):
        self.num_intervals = n

    def transform(self, X):
        """

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, num_atts]
            The training input samples.  If a Pandas data frame is passed,
            the column 0 is extracted

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)

        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        for i in range(num_insts):
            series = X[i, :]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            for n in range(num_atts):
                remaining = frame_length - current_frame_size

                if remaining > 1:
                    frame_sum += series[n]
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    current_frame_size += remaining

                if current_frame_size == frame_length:
                    frames.append(frame_sum / frame_length)
                    current_frame += 1

                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = (1 - remaining)

            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
                frames.append(frame_sum / frame_length)

            data.append(pd.Series(frames))

        dims[0] = data

        return dims


if __name__ == "__main__":
    testPath = "C:\\Users\\ajb\\Dropbox\\Data\\TSCProblems\\Chinatown" \
               "\\Chinatown_TRAIN.ts"
    train_x, train_y = load_ts(testPath)

    print("Correctness testing for PAA using Chinatown")
    #    print("First case used for testing")
    #    print(train_x.iloc[0,0])
    p = PAA()
    print("Test 1: num intervals =1, result should be series mean")
    p.set_num_intervals(1)
    x2 = p.transform(train_x)
    print("Correct mean case 1: =  561.875")
    print("Transform mean case 1: =")
    print(x2.iloc[0, 0])
    print("Test 2: num intervals = series length, series should be unchanged")
    p.set_num_intervals(24)
    x2 = p.transform(train_x)
    print("Before transform: =")
    print(train_x.iloc[0, 0].shape[0])
    print(train_x.iloc[0, 0])
    print("After transform: =")
    print(x2.iloc[0, 0].shape[0])
    print(x2.iloc[0, 0])
    print(
        "Test 3: Integer interval length (length%num intervals =0). num "
        "intervals = 6 (gives length =4). ")
    p.set_num_intervals(6)
    print("Expected output: = 365.25,36.75,293.5,1202.25,994,479.5")
    print("Actual output: =")
    x2 = p.transform(train_x)
    print(x2.iloc[0, 0])

    print(
        "Test 4: Non-integer interval length. num intervals = 5 (gives "
        "length =4.8). ")
    p.set_num_intervals(5)
    print("Expected output: for series[0][0][0] = 313.54")
    print("Actual output: =")
    x2 = p.transform(train_x)
    print(x2.iloc[0, 0])
