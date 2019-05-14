import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sktime.utils import utilities
from sktime.utils.utilities import check_data


class Classifier(BaseEstimator, ClassifierMixin):
    '''
    template classifier to implement the common behaviour, such as prediction of highest probability from
    predict_proba / having a rand attribute for random value sourcing
    ----
    Parameters
    ----
    rand : numpy RandomState
        a random state for sampling random numbers
    ----
    Attributes
    ----
    label_encoder : LabelEncoder
        a label encoder, can be pre-populated
    '''
    def __init__(self,
                 rand = None,
                 ):
        super().__init__()
        self.rand = rand
        self.label_encoder = None

    def predict_proba(self, instances):
        # should be overriden
        raise NotImplementedError('abstract method')

    def predict(self, instances, should_check_data = True):
        '''
        classify instances
        ----
        Parameters
        ----
        instances : panda dataframe
            instances of the dataset
        should_check_data : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        predictions : 1d numpy array
            array of predictions of each instance (class value)
        '''
        if should_check_data:
            check_data(instances)
        distributions = self.predict_proba(instances, should_check_data = False)
        predictions = np.empty((distributions.shape[0]))
        for instance_index in range(0, predictions.shape[0]):
            distribution = distributions[instance_index]
            prediction = utilities.max(distribution, self.rand)
            predictions[instance_index] = prediction
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions
