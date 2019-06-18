
from sktime.classifiers.base import BaseClassifier
from sktime.datasets import load_gunpoint
import sys
import numpy as np
import pandas as pd
import keras
from sklearn.utils.estimator_checks import check_estimator


class BaseDeepLearner(BaseClassifier):

    def build_model(self, input_shape, nb_classes, **kwargs):
        raise NotImplementedError('this is an abstract method')

    def fit(self, X, y, input_checks = True, **kwargs):
        raise NotImplementedError()

    def predict_proba(self, X, input_checks=True, **kwargs):

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        probs = self.model.predict(X, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def convert_y(self, y):
        # taken from kerasclassifier's fit

        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes = np.unique(y)
            y = np.searchsorted(self.classes, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.nb_classes = len(self.classes)

        return keras.utils.to_categorical(y, self.nb_classes)

    def score(self, X, y, **kwargs):
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        y_onehot = self.convert_y(y)

        outputs = self.model.evaluate(X, y_onehot, **kwargs)
        outputs = keras.utils.generic_utils.to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output

def test_basic(network):
    '''
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score,
    ~1min execution for james on gpu
    '''

    print("Start test_basic()\n\n")

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)


    clf = network

    hist = clf.fit(X_train, y_train)
    clf.model.summary()

    print(clf.score(X_test, y_test))
    print("end test_basic()\n\n")

def test_pipeline(network):
    '''
    slightly more generalised test with sktime pipelines
        load data,
        construct pipeline with classifier,
        fit,
        score,
    ~1min execution for james on gpu
    '''

    print("Start test_pipeline()")

    from sktime.pipeline import Pipeline

    # just a simple (not even necessarily good) pipeline for the purposes of testing
    # that the keras network is compatible with that system
    # in fact, the base transform for RISE, so not even technically timeseries
    steps = [
        ('clf', network)
    ]
    clf = Pipeline(steps)

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)

    hist = clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))
    print("end test_pipeline()\n\n")


def test_highLevelsktime(network):
    '''
    truly generalised test with sktime strategies/tasks
        load data, build task
        construct classifier, build strategy
        fit,
        score,
    ~1min execution for james on gpu
    '''

    print("start test_highLevelsktime()\n\n")

    from sktime.highlevel import TSCTask
    from sktime.highlevel import TSCStrategy
    from sklearn.metrics import accuracy_score

    train = load_gunpoint(split='TRAIN')
    test = load_gunpoint(split='TEST')
    task = TSCTask(target='class_val', metadata=train)

    clf = network
    strategy = TSCStrategy(clf)
    strategy.fit(task, train)

    y_pred = strategy.predict(test)
    y_test = test[task.target]
    accuracy_score(y_test, y_pred)

    print("end test_highLevelsktime()\n\n")


def networkTests(network):
    # sklearn compatibility
    # check_estimator(FCN)

    test_basic(network)
    test_pipeline(network)
    test_highLevelsktime(network)


if __name__ == "__main__":

    if len(sys.args > 1):
        network = sys.args[1]
        networkTests(network)


