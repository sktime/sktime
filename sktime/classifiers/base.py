import numpy as np
from sklearn.base import BaseEstimator
from sktime.utils.validation import check_X_y


class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """
    _estimator_type = "classifier"

# predict class labels from predict_proba / label encoder
def predict_from_predict_proba(self, X, input_checks = True):
    if input_checks:
        check_X_y(X)
    distributions = self.predict_proba(X, input_checks = False)
    predictions = np.empty((distributions.shape[0]))
    for instance_index in range(0, predictions.shape[0]):
        distribution = distributions[instance_index]
        prediction = max(distribution, self.rand)
        predictions[instance_index] = prediction
    predictions = self.label_encoder.inverse_transform(predictions)
    return predictions
