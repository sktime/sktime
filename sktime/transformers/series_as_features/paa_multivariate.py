import pandas as pd
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.validation.series_as_features import check_X
from sktime.transformers.series_as_features.dictionary_based._paa import PAA

__author__ = "Vincent Nicholson"

"""
This class is essentially a wrapper for the _paa
transformer but for multivariate data.
Intended for use by ShapeDTW.
"""


class PAA_Multivariate(BaseSeriesAsFeaturesTransformer):

    def __init__(self,
                 num_intervals=8
                 ):
        self.num_intervals = num_intervals
        super(PAA_Multivariate, self).__init__()

    def set_num_intervals(self, n):
        self.num_intervals = n

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

        # Get information about the dataframe
        num_atts = len(X.iloc[0, 0])
        col_names = X.columns

        # Check the parameters are appropriate
        self.check_parameters(num_atts)

        # Create the PAA object
        p = PAA(num_intervals=self.num_intervals)
        p.fit(X)

        # On each dimension, perform PAA
        dataFrames = []
        for x in col_names:
            dataFrames.append(p.transform(pd.DataFrame(X[x])))

        # Combine the dimensions together
        result = pd.concat(dataFrames, axis=1, sort=False)
        result.columns = col_names

        return result

    """
    Function for checking the values of parameters inserted into HOG1D.

    Throws
    ------
    ValueError or TypeError if a parameters input is invalid.
    """
    def check_parameters(self, num_atts):
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError("num_intervals must have the \
                                  value of at least 1")
            if self.num_intervals > num_atts:
                raise ValueError("num_intervals cannot be higher \
                                  than subsequence_length")
        else:
            raise TypeError("num_intervals must be an 'int'. Found '" +
                             type(self.num_intervals).__name__ + "' instead.")
