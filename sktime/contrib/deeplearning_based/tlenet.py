# t-leNettwi-esn
import keras
import numpy as np

import pandas as pd

from sktime.contrib.deeplearning_based.basenetwork import BaseDeepLearner
from sktime.contrib.deeplearning_based.basenetwork import networkTests

class TLENET(BaseDeepLearner):

    def __init__(self, output_directory=None, verbose=False, dim_to_use=0):
        self.output_directory = output_directory
        self.verbose = verbose
        self.warping_ratios = [0.5, 1, 2]
        self.slice_ratio = 0.1
        self.dim_to_use = dim_to_use

        self.nb_epochs = 1000
        self.batch_size = 256

    def slice_data(self, X, y=None, length_sliced=1):
        n = X.shape[0]
        length = X.shape[1]
        n_dim = X.shape[2]  # for MTS

        increase_num = length - length_sliced + 1  # if increase_num =5, it means one ori becomes 5 new instances.
        n_sliced = n * increase_num

        print((n_sliced, length_sliced, n_dim))

        new_x = np.zeros((n_sliced, length_sliced, n_dim))

        for i in range(n):
            for j in range(increase_num):
                new_x[i * increase_num + j, :, :] = X[i, j: j + length_sliced, :]

            # transform the class labels if they're present.
        new_y = None
        if y is not None:
            new_y = np.zeros((n_sliced, self.nb_classes))
            for i in range(n):
                for j in range(increase_num):
                    new_y[i * increase_num + j] = np.int_(y[i].astype(np.float32))

        return new_x, new_y, increase_num

    def window_warping(self, data_x, warping_ratio):
        num_x = data_x.shape[0]
        len_x = data_x.shape[1]
        dim_x = data_x.shape[2]

        x = np.arange(0, len_x, warping_ratio)
        xp = np.arange(0, len_x)

        new_length = len(np.interp(x, xp, data_x[0, :, 0]))

        warped_series = np.zeros((num_x, new_length, dim_x), dtype=np.float64)

        for i in range(num_x):
            for j in range(dim_x):
                warped_series[i, :, j] = np.interp(x, xp, data_x[i, :, j])

        return warped_series

    def build_model(self, input_shape, nb_classes, **kwargs):
        input_layer = keras.layers.Input(input_shape)

        conv_1 = keras.layers.Conv1D(filters=5, kernel_size=5, activation='relu', padding='same')(input_layer)
        conv_1 = keras.layers.MaxPool1D(pool_size=2)(conv_1)

        conv_2 = keras.layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding='same')(conv_1)
        conv_2 = keras.layers.MaxPool1D(pool_size=4)(conv_2)

        # they did not mention the number of hidden units in the fully-connected layer
        # so we took the lenet they referenced 

        flatten_layer = keras.layers.Flatten()(conv_2)
        fully_connected_layer = keras.layers.Dense(500, activation='relu')(flatten_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer=keras.optimizers.Adam(lr=0.01, decay=0.005),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # file_path = self.output_directory+'best_model.hdf5'

        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        # save_best_only=True)

        self.callbacks = []

        return model

    def pre_processing(self, X, y=None):
        length_ratio = int(self.slice_ratio * X.shape[1])

        x_augmented = []  # list of the augmented as well as the original data

        if y is not None:
            y_augmented = []

        # data augmentation using WW
        for warping_ratio in self.warping_ratios:
            x_augmented.append(self.window_warping(X, warping_ratio))

            if y is not None:
                y_augmented.append(y)


        increase_nums = []

        # data augmentation using WS
        for i in range(0, len(x_augmented)):
            x_augmented[i], y_train_augmented_i, increase_num = self.slice_data(x_augmented[i], y, length_ratio)
            print("inc num",increase_num)
            if y is not None:
                y_augmented[i] = y_train_augmented_i

            increase_nums.append(increase_num)

        tot_increase_num = np.array(increase_nums).sum()

        new_x = np.zeros((X.shape[0] * tot_increase_num, length_ratio, X.shape[2]))

        # merge the list of augmented data
        idx = 0
        for i in range(X.shape[0]):
            for j in range(len(increase_nums)):
                increase_num = increase_nums[j]
                new_x[idx:idx + increase_num, :, :] = x_augmented[j][i * increase_num:(i + 1) * increase_num, :, :]
                idx += increase_num

        # merge y if its not None.
        new_y = None
        if y is not None:
            new_y = np.zeros((y.shape[0] * tot_increase_num, y.shape[1]))
            idx = 0
            for i in range(X.shape[0]):
                for j in range(len(increase_nums)):
                    increase_num = increase_nums[j]
                    new_y[idx:idx + increase_num, :] = y_augmented[j][i * increase_num:(i + 1) * increase_num, :]
                    idx += increase_num
					
					
        return new_x, new_y, tot_increase_num


    def fit(self, X, y, input_checks=True, **kwargs):
        # check and convert input to a univariate Numpy array
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")
        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        y = self.convert_y(y)

        self.nb_classes = y.shape[1]
        n = X.shape[0]  # num cases
        m = X.shape[1]  # series length

        # limit the number of augmented time series if series too long or too many
        if m > 500 or n > 2000:
            self.warping_ratios = [1]
            self.slice_ratio = 0.9
        # increase the slice if series too short
        if m * self.slice_ratio < 8:
            self.slice_ratio = 8 / m

        X, y, tot_increase_num = self.pre_processing(X, y)
        print(y.shape)

        print('Total increased number for each MTS: ', tot_increase_num)

        input_shape = X.shape[1:]
        model = self.build_model(input_shape, self.nb_classes)

        self.hist = model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs,
                         verbose=self.verbose, callbacks=self.callbacks)


    def predict_proba(self, X, input_checks=True, **kwargs):
        # preprocess test.
        # check and convert input to a univariate Numpy array
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")
        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        X, _, tot_increase_num = self.pre_processing(X)
        print(X.shape)

        # predict some stuff based on the keras.
        preds = self.hist.predict(X, batch_size=self.batch_size)

        y_predicted = []
        test_num_batch = int(X.shape[0]/tot_increase_num)

        ##TODO: could fix this to be an array literal.
        for i in range(test_num_batch):
            y_predicted.append(np.average(preds[i*tot_increase_num: ((i+1)*tot_increase_num)-1], axis=0))

        y_pred = np.array(y_predicted)

        keras.backend.clear_session()

        return y_pred

if __name__ == "__main__":
    networkTests(TLENET())
