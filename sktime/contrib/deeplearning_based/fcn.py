#FCN
import keras
import numpy as np
import pandas as pd

from sktime.utils.validation import check_X_y
from sktime.contrib.deeplearning_based.basenetwork import BaseDeepLearner
from sktime.contrib.deeplearning_based.basenetwork import networkTests


class FCN(BaseDeepLearner):

    def __init__(self, dim_to_use=0, rand_seed=0, verbose=False):
        self.verbose = verbose
        self.dim_to_use = dim_to_use

        # calced in fit
        self.classes = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # predefined
        self.nb_epochs = 2000
        self.batch_size = 16
        self.callbacks = None

        self.rand_seed = rand_seed
        self.random_state = np.random.RandomState(self.rand_seed)

    def build_model(self, input_shape, nb_classes, **kwargs):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.normalization.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.normalization.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        #file_path = self.output_directory + 'best_model.hdf5'
        #model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                   save_best_only=True)
        #self.callbacks = [reduce_lr, model_checkpoint]
        self.callbacks = [reduce_lr]

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

        self.batch_size = int(min(X.shape[0] / 10, self.batch_size))

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, callbacks=self.callbacks)

if __name__ == '__main__':
    networkTests(FCN())