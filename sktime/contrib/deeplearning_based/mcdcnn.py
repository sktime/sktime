# Multi channel deep convolutional neural network, adapted from the implementation from Fawaz et. al
# https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py
#
# Network originally proposed by:
#
# @inproceedings{zheng2014time,
#   title={Time series classification using multi-channels deep convolutional neural networks},
#   author={Zheng, Yi and Liu, Qi and Chen, Enhong and Ge, Yong and Zhao, J Leon},
#   booktitle={International Conference on Web-Age Information Management},
#   pages={298--310},
#   year={2014},
#   organization={Springer}
# }

__author__ = "James Large"

import keras
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sktime.utils.validation import check_X_y
from sktime.contrib.deeplearning_based.basenetwork import BaseDeepLearner
from sktime.contrib.deeplearning_based.basenetwork import networkTests

class MCDCNN(BaseDeepLearner):

    def __init__(self, dim_to_use=0, rand_seed=0, verbose=False):
        self.verbose = verbose
        self.dim_to_use = dim_to_use

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # predefined
        self.nb_epochs = 120
        self.batch_size = 16

        self.rand_seed = rand_seed

    def build_model(self, input_shape, nb_classes, **kwargs):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60: # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t,1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732,activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=0.0005),
                      metrics=['accuracy'])

        #file_path = self.output_directory + 'best_model.hdf5'
        #model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
        #                                                   save_best_only=True)
        #self.callbacks = [model_checkpoint]
        self.callbacks = []

        return model

    def prepare_input(self,x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:,:,i:i+1])

        return  new_x

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


        x_train, x_val, y_train, y_val = \
            train_test_split(X, y, test_size=0.33)

        y_train_onehot = self.convert_y(y_train)
        y_val_onehot = self.convert_y(y_val)

        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(x_train, y_train_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val_onehot), callbacks=self.callbacks)

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

        x_test = self.prepare_input(X)

        probs = self.model.predict(x_test, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

if __name__ == '__main__':
    networkTests(MCDCNN())

