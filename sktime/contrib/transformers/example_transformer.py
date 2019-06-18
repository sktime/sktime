import numpy as np
import pandas as pd
from sktime.base import BaseTransformer


__all__ = ['Padder']
__author__ = ["Tony Bagnall"]


class Padder(BaseTransformer):
    """
        Padding transformer.

    Parameters
    ----------
    dim_to_use:
    """

    def __init__(self, dim_to_use=None):
        self.dim_to_use=dim_to_use

#No need for fit for Padding transformer




    def transform(self, X, y=None):
        """
        Transform X,

        Parameters
        ----------
        X : nested pandas DataFrame, multi dimensional, with series of
different lengths
        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        #So do we insist on Pandas or allow for 2d numpy array?
        # If we do, how is that defined? Cannot
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")
        num_cases, num_dimensions = X.shape

        if self.dim_to_use is None:   #Do all dimensions
            start=0
            end=num_dimensions
        else:
            start=self.dim_to_use
            end=self.dim_to_use+1
#Find maximum length over all dimensions considered
        for dim:
            if isinstance(X.iloc[0,self.dim_to_use], pd.Series):
            else:
                raise TypeError("Elements of the data frame should be series objects")

                X = np.asarray([a.values for a in X.iloc[:,self.dim_to_use]])
