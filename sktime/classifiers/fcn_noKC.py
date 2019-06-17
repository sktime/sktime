import keras
import numpy as np
import pandas as pd

########   sklearn
from sklearn.utils.estimator_checks import check_estimator

#########  sktime
from sktime.classifiers.base import BaseClassifier
from sktime.utils.validation import check_X_y

# data loading
from sktime.datasets import load_gunpoint


class FCN(BaseClassifier):

    def __init__(self,
                 output_directory=None,
                 verbose=False,
                 dim_to_use=0):

        self.verbose = verbose

        self.dim_to_use = dim_to_use

        self.output_directory = output_directory
        self.filters = [128, 256, 128]
        self.kernel_sizes = [8, 5, 3]
        self.nb_epochs = 1500
        self.depth = 3

        #calced in fit
        self.classes = None
        self.nb_classes = -1
        self.batch_size = -1
        self.input_shape = None
        self.model = None

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Layer(input_shape=self.input_shape))

        for i in range(self.depth):

            model.add(keras.layers.Conv1D(self.filters[i], self.kernel_sizes[i],
                                          padding="same"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(self.nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        return model

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
        self.batch_size = min(X.shape[0] // 10, 16)
        self.input_shape = X.shape[1:]

        self.model = self.build_model()

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        if self.output_directory is None:
            callbacks = [reduce_lr]
        else:
            file_path = self.output_directory + 'best_model.hdf5'
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                               save_best_only=True)
            callbacks = [reduce_lr, model_checkpoint]

        self.model.fit(X, y_onehot, epochs=self.nb_epochs, batch_size=self.batch_size, callbacks=callbacks, verbose=self.verbose)

    def predict_proba(self, X, **kwargs):
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

    def predict(self, X, **kwargs):
        return np.argmax(self.predict_proba(X), axis=1)

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

def test_basic():
    '''
    just a super basic test with gunpoint, ~1min execution
    '''

    '''
    dataset = 'GunPoint'
    path_to_train_data = 'C:/Univariate2018_ts/' + \
                         dataset + '/' + dataset + '_TRAIN.ts'
    train_x, train_y = load_from_tsfile_to_dataframe(path_to_train_data)

    # load the test data
    path_to_test_data = 'C:/Univariate2018_ts/' + \
                        dataset + '/' + dataset + '_TEST.ts'
    test_x, test_y = load_from_tsfile_to_dataframe(path_to_test_data)
    '''

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)


    clf = FCN()

    hist = clf.fit(X_train, y_train)
    clf.model.summary()

    print(clf.score(X_test, y_test))

if __name__ == "__main__":

    #check_estimator(FCN)

    test_basic()            #working
    #test_sklearnPipeline() #broken, missing imports
