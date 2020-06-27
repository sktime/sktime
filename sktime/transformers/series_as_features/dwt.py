import numpy as np
import pandas as pd
import math
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X

__author__ = "Vincent Nicholson"


class DWTTransformer(BaseSeriesAsFeaturesTransformer):

    def __init__(self, num_levels=3):
        self.num_levels = num_levels
        super(DWTTransformer, self).__init__()

    """
    Parameters
    ----------
    X : a pandas dataframe of shape = [n_samples, num_dims]
        The training input samples.

    Returns
    -------
    dims: a pandas data frame of shape = [n_samples, num_dims]
    """
    def transform(self, X, y=None):

        # Check the data
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False)
        
        self.check_parameters()

        # Get information about the dataframe
        col_names = X.columns

        df = pd.DataFrame()
        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = tabularize(pd.DataFrame(X[x]), return_array=True)

            transformedData = self.extract_wavelet_coefficients(arr)

            # Convert to a numpy array
            transformedData = np.asarray(transformedData)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            df[x] = colToAdd

        return df

    """
    Function to extract the wavelet coefficients of a 2d array of time series.

    The coefficients correspond to the wavelet coefficients from levels 1 to
    num_levels followed by the approximation coefficients of the highest level.
    """
    def extract_wavelet_coefficients(self, data):
        num_levels = self.num_levels
        res = []

        for x in data:
            if num_levels == 0:
                res.append(x)
            else:
                coeffs = []
                current = x
                approx = None
                for lev in range(num_levels):
                    approx = self.get_approx_coefficients(current)
                    wav_coeffs = self.get_wavelet_coefficients(current)
                    current = approx
                    wav_coeffs.reverse()
                    coeffs.extend(wav_coeffs)
                approx.reverse()
                coeffs.extend(approx)
                coeffs.reverse()
                res.append(coeffs)

        return res

    """
    Function for checking the values of parameters inserted into DWT.

    Throws
    ------
    ValueError or TypeError if a parameters input is invalid.
    """
    def check_parameters(self):
        if isinstance(self.num_levels, int):
            if self.num_levels <= -1:
                raise ValueError("num_levels must have the value" +
                                  "of at least 0")
        else:
            raise TypeError("num_levels must be an 'int'. Found" +
                            "'" + type(self.num_levels).__name__ +
                            "' instead.")

    """
    Function to get the approximate coefficients at a given level.
    """
    def get_approx_coefficients(self, arr):
        new = []
        if len(arr) == 1:
            return [arr[0]]
        for x in range(math.floor(len(arr)/2)):
            new.append((arr[2*x]+arr[2*x+1])/math.sqrt(2))
        return new

    """
    Function to get the wavelet coefficients at a given level.
    """
    def get_wavelet_coefficients(self, arr):
        new = []
        # if length is 1, just return the list back
        if len(arr) == 1:
            return [arr[0]]
        for x in range(math.floor(len(arr)/2)):
            new.append((arr[2*x]-arr[2*x+1])/math.sqrt(2))
        return new
