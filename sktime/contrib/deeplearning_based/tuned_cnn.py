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
from sktime.contrib.deeplearning_based.basenetwork import BaseDeepLearner
from sktime.contrib.deeplearning_based.basenetwork import networkTests

from sklearn.model_selection import GridSearchCV
from sktime.datasets import load_italy_power_demand

from sktime.contrib.deeplearning_based.dl4tsc.cnn import CNN

class CNN_Tunable(BaseDeepLearner):

    def __init__(self, dim_to_use=0, rand_seed=0, verbose=False,
                 nb_epochs=2000,
                 batch_size=16,
                 kernel_size=7,
                 avg_pool_size=3,
                 nb_conv_layers=2,
                 filter_sizes=[6,12]):
        self.verbose = verbose
        self.dim_to_use = dim_to_use

        self.callbacks = []

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # TUNABLE PARAMETERS
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.nb_conv_layers = nb_conv_layers
        self.filter_sizes = filter_sizes

        self.rand_seed = rand_seed
        self.random_state = np.random.RandomState(self.rand_seed)


    def build_model(self, input_shape, nb_classes, **kwargs):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        if len(self.filter_sizes) > self.nb_conv_layers:
            self.filter_sizes = self.filter_sizes[:self.nb_conv_layers]
        elif len(self.filter_sizes) < self.nb_conv_layers:
            self.filter_sizes = self.filter_sizes + [self.filter_sizes[-1]] * (self.nb_conv_layers - len(self.filter_sizes))

        conv = keras.layers.Conv1D(filters=self.filter_sizes[0],
                                    kernel_size=self.kernel_size,
                                    padding=padding,
                                    activation='sigmoid')(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        for i in range(1, self.nb_conv_layers):
            conv = keras.layers.Conv1D(filters=self.filter_sizes[i],
                                        kernel_size=self.kernel_size,
                                        padding=padding,
                                        activation='sigmoid')(conv)
            conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        flatten_layer = keras.layers.Flatten()(conv)
        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        #file_path = self.output_directory + 'best_model.hdf5'
        #model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                   save_best_only=True)
        #self.callbacks = [model_checkpoint]

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        if input_checks:
            check_X_y(X, y)

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
        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape, self.nb_classes, **kwargs)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, callbacks=self.callbacks, **kwargs)


if __name__ == '__main__':
    model = CNN()
    # model = CNN_Tunable()

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    param_grid = dict(
        #nb_epochs=[5, 10],
        #batch_size=[8, 16],
        kernel_size=[3, 7],
        avg_pool_size=[2, 3],
        nb_conv_layers=[1, 2],
        filter_sizes=[[6, 12], [4, 8]]
    )
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    # X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)
    #
    # c = CNN_Tunable()
    # hist = c.fit(X_train, y_train, nb_epochs=10)
    # print(c.score(X_test, y_test))

    #networkTests(CNN_Tunable())