from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from sktime.utils import comparison
from sktime.utils.validation import check_X, check_X_y


class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """
    _estimator_type = "classifier"
    label_encoder = None
    random_state = None

    def fit(self, X, y, input_checks=True):
        raise NotImplementedError('this is an abstract method')

    def predict_proba(self, X, input_checks=True):
        raise NotImplementedError('this is an abstract method')

    def predict(self, X, input_checks=True):
        """
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
        """
        if input_checks:
            check_X(X)
        distributions = self.predict_proba(X, input_checks=False)
        predictions = []
        for instance_index in range(0, X.shape[0]):
            distribution = distributions[instance_index]
            prediction = comparison.arg_max(distribution, self.random_state)
            predictions.append(prediction)
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

    def score(self, X, y):
        check_X_y(X, y)
        predictions = self.predict(X)
        acc = accuracy_score(y, predictions, normalize=True)
        return acc
