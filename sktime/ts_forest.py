from .base import BaseClassifier
import numpy as np


class TSForest(BaseClassifier):
    """
    Implementation of Deng's Time Series Forest:

    Reference
    ---------

    @article{deng13forest,
    author = {H. Deng and G. Runger and E. Tuv and M. Vladimir},
    title = {A time series forest for classification and feature extraction},
    journal = {Information Sciences},
    volume = {239},
    year = {2013}
    """

    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y):
        """

        Requires equal length series

        :param X:
        :param y:
        :return:
        """


        n_intervals = np.sqrt()
        n_starting_points = np.sqrt()

        n_features = np.sqrt(X.shape[1])



        self.is_fitted_ = True
        return self


    def predict(self, X, y):
        pass

    def predict_proba(self, X, y):
        pass

