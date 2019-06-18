# dummy transform
import numpy as np
import pandas as pd
from sktime.transformers.base import BaseTransformer


class DummyTransformer(BaseTransformer):

    def __init__(self, check_input=True):
        self.check_input = check_input
        self.num_cases = None
        self.num_dimensions = None

    def fit(self, X, y=None):

        if self.check_input:
            pass

        self.num_cases, self.num_dimensions = X.shape

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be a pandas dataframe containing Series objects")

        return X


if __name__ == '__main__':
    location = r'C:\Users\Jeremy\Desktop\Beef\Beef_TRAIN.txt'
    X = pd.read_csv(location, header=None)

    print(X)
    transformer = DummyTransformer()
    Xt = transformer.transform(X)

    print(Xt)
