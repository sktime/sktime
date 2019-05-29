import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sktime.utils.validation import check_X_y
from sktime.utils import comparison

class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """
    _estimator_type = "classifier"
    label_encoder = None
    random_state = None

    def fit(self, X, y):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError('this is an abstract method')

    def predict(self, X):
        '''
        classify instances
        ----
        Parameters
        ----
        X : panda dataframe
            instances of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        predictions : 1d numpy array
            array of predictions of each instance (class value)
        '''
        if input_checks:
            check_X_y(X)
        distributions = self.predict_proba(X, input_checks = False)
        predictions = []
        for instance_index in range(0, X.shape[0]):
            distribution = distributions[instance_index]
            prediction = comparison.arg_max(distribution, self.random_state)
            predictions.append(prediction)
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        acc = accuracy_score(y, predictions, normalize = True)
        return acc
