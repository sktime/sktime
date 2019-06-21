# Time convolutional neural network, adapted from the implementation from Fawaz et. al
# https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py
#
# Default parameters (without tuning) corresponds to the exact setup defined cnn
#
#
# Network originally proposed by:
#
# @article{zhao2017convolutional,
#   title={Convolutional neural networks for time series classification},
#   author={Zhao, Bendong and Lu, Huanzhang and Chen, Shangfeng and Liu, Junliang and Wu, Dongya},
#   journal={Journal of Systems Engineering and Electronics},
#   volume={28},
#   number={1},
#   pages={162--169},
#   year={2017},
#   publisher={BIAI}
# }

__author__ = "James Large"

import keras
import numpy as np
import pandas as pd

from sktime.utils.validation import check_X_y
from sktime.classifiers.base import BaseClassifier
from sktime.contrib.deeplearning_based.basenetwork import networkTests

from sklearn.model_selection import GridSearchCV
from sktime.datasets import load_italy_power_demand

from sktime.contrib.deeplearning_based.dl4tsc.cnn import CNN


class CNN_Tunable(BaseClassifier):

    def __init__(self, dim_to_use=0, rand_seed=0, verbose=False, n_jobs=1,
                 param_grid=dict(
                     kernel_size=[3, 7],
                     avg_pool_size=[2, 3],
                     nb_conv_layers=[1, 2],
                 )):
        self.verbose = verbose
        self.dim_to_use = dim_to_use
        self.n_jobs = n_jobs

        self.param_grid = param_grid
        self.grid_result = None
        self.grid = None
        self.tuned_params = None

        self.base_model = CNN()
        # todo make decisions on wrapping each network
        #  separately or one by one, etc.
        self.tuned_model = None

        self.rand_seed = rand_seed
        self.random_state = np.random.RandomState(self.rand_seed)

    def fit(self, X, y, input_checks=True, **kwargs):
        self.grid = GridSearchCV(estimator=self.base_model,
                                 param_grid=self.param_grid,
                                 n_jobs=self.n_jobs)
        self.grid_result = self.grid.fit(X, y)

        self.tuned_model = self.grid.best_estimator_
        self.tuned_params = self.grid.best_params_

        return self

    def predict_proba(self, X, input_checks=True):
        return self.grid.predict_proba(X)

    def get_tuned_model(self):
        return self.tuned_model

    def get_tuned_params(self):
        return self.tuned_params

    def print_search_summary(self):
        print("Best: %f using %s" % (self.grid_result.best_score_, self.grid_result.best_params_))
        means = self.grid_result.cv_results_['mean_test_score']
        stds = self.grid_result.cv_results_['std_test_score']
        params = self.grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    networkTests(CNN_Tunable())
