# Base class for the Keras neural networks adapted from Fawaz et. al
# https://github.com/hfawaz/dl-4-tsc
#
# @article{fawaz2019deep,
#   title={Deep learning for time series classification: a review},
#   author={Fawaz, Hassan Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
#   journal={Data Mining and Knowledge Discovery},
#   pages={1--47},
#   year={2019},
#   publisher={Springer}
# }
#
# File also contains some simple unit-esque tests for the networks and
# their compatibility wit hthe rest of the package,
# and experiments for confirming accurate reproduction
#
# todo proper unit tests
# todo confirm compaitbility of class bales especially between networks and rest of sktime


__author__ = "James Large, Aaron Bostrom"

import sys
import numpy as np
import pandas as pd
import keras
import gc

from sktime.classifiers.base import BaseClassifier
from sktime.datasets import load_italy_power_demand
from sktime.utils.load_data import load_from_tsfile_to_dataframe

from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class BaseDeepLearner(BaseClassifier):
    classes_ = None
    nb_classes = None

    def build_model(self, input_shape, nb_classes, **kwargs):
        raise NotImplementedError('this is an abstract method')

    def fit(self, X, y, input_checks=True, **kwargs):
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
        ### taken from kerasclassifier's fit
        # y = np.array(y)
        # if len(y.shape) == 2 and y.shape[1] > 1:
        #    self.classes_ = np.arange(y.shape[1])
        # elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
        #    self.classes_ = np.unique(y)
        #    y = np.searchsorted(self.classes_, y)
        # else:
        #    raise ValueError('Invalid shape for y: ' + str(y.shape))
        # self.nb_classes = len(self.classes_)
        #
        # return keras.utils.to_categorical(y, self.nb_classes)

        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)

        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.nb_classes = len(self.classes_)

        y = y.reshape(len(y), 1)
        y = self.onehot_encoder.fit_transform(y)

        return y


def test_basic_univariate(network):
    '''
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    '''

    print("Start test_basic()\n\n")

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("end test_basic()\n\n")


def test_pipeline(network):
    '''
    slightly more generalised test with sktime pipelines
        load data,
        construct pipeline with classifier,
        fit,
        score
    '''

    print("Start test_pipeline()")

    from sktime.pipeline import Pipeline

    # just a simple (useless) pipeline for the purposes of testing
    # that the keras network is compatible with that system
    steps = [
        ('clf', network)
    ]
    clf = Pipeline(steps)

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = clf.fit(X_train[:10], y_train[:10])

    print(clf.score(X_test[:10], y_test[:10]))
    print("end test_pipeline()\n\n")


def test_highLevelsktime(network):
    '''
    truly generalised test with sktime tasks/strategies
        load data, build task
        construct classifier, build strategy
        fit,
        score
    '''

    print("start test_highLevelsktime()\n\n")

    from sktime.highlevel import TSCTask
    from sktime.highlevel import TSCStrategy
    from sklearn.metrics import accuracy_score

    train = load_italy_power_demand(split='TRAIN')
    test = load_italy_power_demand(split='TEST')
    task = TSCTask(target='class_val', metadata=train)

    strategy = TSCStrategy(network)
    strategy.fit(task, train.iloc[:10])

    y_pred = strategy.predict(test.iloc[:10])
    y_test = test[task.target]
    print(accuracy_score(y_test, y_pred))

    print("end test_highLevelsktime()\n\n")


def test_basic_multivariate(network):
    '''
    just a super basic test with basicmotions,
        load data,
        construct classifier,
        fit,
        score
    '''
    print("Start test_multivariate()\n\n")

    X_train, y_train = load_from_tsfile_to_dataframe(
        'Z:/sktimeData/Multivariate2018_ts/BasicMotions/BasicMotions_TRAIN.ts')
    X_test, y_test = load_from_tsfile_to_dataframe(
        'Z:/sktimeData/Multivariate2018_ts/BasicMotions/BasicMotions_TEST.ts')

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("end test_multivariate()\n\n")


def test_network(network):
    # sklearn compatibility
    # check_estimator(FCN)

    test_basic_univariate(network)
    test_basic_multivariate(network)
    test_pipeline(network)
    test_highLevelsktime(network)


def test_all_networks_all_tests():
    import sktime.contrib.deeplearning_based.dl4tsc.cnn as cnn
    import sktime.contrib.deeplearning_based.dl4tsc.encoder as encoder
    import sktime.contrib.deeplearning_based.dl4tsc.fcn as fcn
    import sktime.contrib.deeplearning_based.dl4tsc.mcdcnn as mcdcnn
    import sktime.contrib.deeplearning_based.dl4tsc.mcnn as mcnn
    import sktime.contrib.deeplearning_based.dl4tsc.mlp as mlp
    import sktime.contrib.deeplearning_based.dl4tsc.resnet as resnet
    import sktime.contrib.deeplearning_based.dl4tsc.tlenet as tlenet
    import sktime.contrib.deeplearning_based.dl4tsc.twiesn as twiesn
    import sktime.contrib.deeplearning_based.tuned_cnn as tuned_cnn

    networks = [cnn.CNN(),
                encoder.Encoder(),
                fcn.FCN(),
                mcdcnn.MCDCNN(),
                mcnn.MCNN(),
                mlp.MLP(),
                resnet.ResNet(),
                tlenet.TLENET(),
                twiesn.TWIESN(),
                tuned_cnn.Tuned_CNN(),
                ]

    for network in networks:
        print('\t\t' + network.__class__.__name__ + ' testing started')
        test_network(network)
        print('\t\t' + network.__class__.__name__ + ' testing finished')


def comparisonExperiments():
    data_dir = "C:/Univariate2018_ts/"
    res_dir = "C:/JamesLPHD/sktimeStuff/InitialComparisonResults/"

    complete_classifiers = [
        "dl4tsc_cnn",
        "dl4tsc_encoder",
        "dl4tsc_fcn",
        "dl4tsc_mcdcnn",
        # "dl4tsc_mcnn",
        "dl4tsc_mlp",
        "dl4tsc_resnet",
        # "dl4tsc_tlenet",
        # "dl4tsc_twiesn",
    ]

    small_datasets = [
        "Beef",
        "Car",
        "Coffee",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "Fish",
        "GunPoint",
        "ItalyPowerDemand",
        "MoteStrain",
        "OliveOil",
        "Plane",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SyntheticControl",
        "Trace",
        "TwoLeadECG",
    ]

    num_folds = 30

    import sktime.contrib.experiments as exp

    for f in range(num_folds):
        for d in small_datasets:
            for c in complete_classifiers:
                print(c, d, f)
                try:
                    exp.run_experiment(data_dir, res_dir, c, d, f)
                    gc.collect()
                    keras.backend.clear_session()
                except:
                    print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    # comparisonExperiments()
    test_all_networks_all_tests()
