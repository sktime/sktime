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
from sktime.contrib.deeplearning_based.basenetwork import BaseDeepLearner
from sktime.contrib.deeplearning_based.basenetwork import networkTests

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sktime.contrib.deeplearning_based.dl4tsc.cnn import CNN


class CNN_Tunable(BaseDeepLearner):

    def __init__(self, dim_to_use=0, rand_seed=0, verbose=False, n_jobs=1,
                 param_grid=dict(
                     kernel_size=[3, 7],
                     avg_pool_size=[2, 3],
                     nb_conv_layers=[1, 2],
                 ),
                 search_method='grid',
                 cv_folds=5):
        self.verbose = verbose
        self.dim_to_use = dim_to_use
        self.rand_seed = rand_seed
        self.random_state = np.random.RandomState(self.rand_seed)

        self.base_model = CNN()
        # todo make decisions on wrapping each network
        #  separately or one by one, etc.

        # search parameters
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.search_method = search_method
        self.n_jobs = n_jobs

        # search results (computed in fit)
        self.grid_history = None
        self.grid = None
        self.model = None  # the best _keras model_, not the sktime classifier object
        self.tuned_params = None

    def build_model(self, input_shape, nb_classes, **kwargs):
        if self.tuned_params is not None:
            return self.base_model.build_model(input_shape, nb_classes, **kwargs)
        else:
            return self.base_model.build_model(input_shape, nb_classes, self.tuned_params)

    def fit(self, X, y, input_checks=True, **kwargs):
        if self.search_method is 'grid':
            self.grid = GridSearchCV(estimator=self.base_model,
                                     param_grid=self.param_grid,
                                     cv=self.cv_folds,
                                     n_jobs=self.n_jobs)
        elif self.search_method is 'random':
            self.grid = RandomizedSearchCV(estimator=self.base_model,
                                           param_grid=self.param_grid,
                                           cv=self.cv_folds,
                                           n_jobs=self.n_jobs)
        else:
            # todo expand, give options etc
            raise Exception('Unrecognised search method provided: {}'.format(self.search_method))

        self.grid_history = self.grid.fit(X, y, refit=True)
        self.model = self.grid.best_estimator_.model
        self.tuned_params = self.grid.best_params_

        # copying data-wrangling info up
        self.label_encoder = self.grid.best_estimator_.label_encoder#
        self.classes_ = self.grid.best_estimator_.classes_
        self.nb_classes = self.grid.best_estimator_.nb_classes

        if self.verbose:
            self.print_search_summary()

        return self

    def get_tuned_model(self):
        return self.model

    def get_tuned_params(self):
        return self.tuned_params

    def print_search_summary(self):
        print("Best: %f using %s" % (self.grid_history.best_score_, self.grid_history.best_params_))
        means = self.grid_history.cv_results_['mean_test_score']
        stds = self.grid_history.cv_results_['std_test_score']
        params = self.grid_history.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    # simple, small, fast search for testing. default nb_epochs = 2000
    param_grid = dict(nb_epochs=[5, 10])
    networkTests(CNN_Tunable(param_grid=param_grid, cv_folds=2))
