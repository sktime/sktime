from sktime.transformers.base import BaseTransformer

from sktime.utils.transformations import tabularise
import pandas as pd

class PandasToNumpy(BaseTransformer):

    def __init__(self,
                 cls = None,
                 unpack_train = True,
                 unpack_test = True):
        self.cls = cls
        self.unpack_train = unpack_train
        self.unpack_test = unpack_test

    def transform(self, X, y=None):
        if self.unpack_train and isinstance(X, pd.DataFrame): X = tabularise(X, return_array=True)
        return X
