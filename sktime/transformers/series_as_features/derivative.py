import numpy as np
import pandas as pd
from sktime.transformers.series_as_features.base \
    import BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X


class DerivativeTransformer(BaseSeriesAsFeaturesTransformer):

    """
    Function to calculate the derivative of a time series.
    Does the same formula as dDTW.

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
        num_insts = X.shape[0]
        col_names = X.columns

        df = pd.DataFrame()

        for x in col_names:
            # Convert one of the columns in the dataframe to a numpy array
            arr = tabularize(pd.DataFrame(X[x]), return_array=True)

            # Calculate the derivative of each time series.
            transformedData = []
            for y in range(num_insts):
                inst = [arr[y][i]-arr[y][(i+1)] for i in range(num_atts-1)]
                transformedData.append(inst)

            # Convert to numpy array
            transformedData = np.asarray(transformedData)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            df[x] = colToAdd

        return df
