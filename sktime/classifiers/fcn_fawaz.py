import keras
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier

########   sklearn
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator

#########sktime
# orchistration
#from sktime.highlevel import TSCTask           # both not importing correctly. missing some forcasting import
#from sktime.highlevel import TSCStrategy

# data loading
from sktime.datasets import load_gunpoint
from sktime.utils.load_data import load_from_tsfile_to_dataframe

# class Residual(Layer):
#
#     def __init__(self, filters, kernel_sizes, **kwargs):
#         super(Residual, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_sizes = kernel_sizes
#
#     def build(self, input_shape):
#         for i in range(1):
#             self.add_weight(name='kernel'+str(i),
#                             shape=(self.kernel_sizes[i], input_shape[2], self.filters),
#                             initializer='uniform',
#                             trainable=True)
#         super(Residual, self).build(input_shape)
#
#     def call(self, x, **kwargs):
#         # the residual block using Keras functional API
#         first_layer = x
#         # conv 1
#         x = keras.layers.Conv1D(self.filters, self.kernel_sizes[0],
#                                 padding="same")(first_layer)
#         # x = keras.layers.BatchNormalization()(x)
#         # x = keras.layers.Activation("relu")(x)
#         #
#         # # conv 2
#         # x = keras.layers.Conv1D(self.filters, self.kernel_sizes[1],
#         #                         padding="same")(x)
#         # x = keras.layers.BatchNormalization()(x)
#         # x = keras.layers.Activation("relu")(x)
#         #
#         # # conv 3
#         # x = keras.layers.Conv1D(self.filters, self.kernel_sizes[2],
#         #                         padding="same")(x)
#         # x = keras.layers.BatchNormalization()(x)
#         #
#         # # expand channels for the sum
#         # shortcut = keras.layers.Conv1D(self.filters, kernel_size=1,
#         #                                padding="same")(first_layer)
#         # shortcut = keras.layers.BatchNormalization()(shortcut)
#         #
#         # x = keras.layers.Add()([shortcut, x])
#         # x = keras.layers.Activation("relu")(x)
#
#         return x

# def compute_output_shape(self, input_shape):
#     return (input_shape[0], input_shape[1], self.filters)


class FCN(KerasClassifier):

    def __init__(self, nb_classes, batch_size, input_shape, verbose=False, dim_to_use=0):
        self.verbose = verbose
        self.dim_to_use = dim_to_use

        self.output_directory = 'C:/JamesLPHD/sktimeStuff/'
        self.filters = [128, 256, 128]
        self.kernel_sizes = [8, 5, 3]
        self.nb_epochs = 1500
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.depth = 3

        def build_fn(inputshape=self.input_shape, nbclasses=self.nb_classes):
            model = keras.models.Sequential()
            model.add(keras.layers.Layer(input_shape=inputshape))

            for i in range(self.depth):

                model.add(keras.layers.Conv1D(self.filters[i], self.kernel_sizes[i],
                                              padding="same"))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Activation('relu'))

            model.add(keras.layers.GlobalAveragePooling1D())
            model.add(keras.layers.Dense(nbclasses, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

            return model

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        callbacks = [reduce_lr, model_checkpoint]

        super(FCN, self).__init__(build_fn=build_fn, epochs=self.nb_epochs,
                                  batch_size=self.batch_size,
                                  verbose=self.verbose,
                                  callbacks=callbacks)

    def fit(self, X, y, **kwargs):
        """
        Build a neural network for the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples. If a Pandas data frame is passed, the column _dim_to_use is extracted
        y : array-like, shape = [n_samples, n_outputs]
            The class labels.
        Returns
        -------
        self : object
        """
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        return super(FCN, self).fit(X, y)

    def predict_proba(self, X, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples. If a Pandas data frame is passed, the column _dim_to_use is extracted
        Local variables
        ----------
        n_samps     : number of cases to classify
        num_atts    : number of attributes in X, must match _num_atts determined in fit
        Returns
        -------
        output : 2D array of probabilities,
        """

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        return super(FCN, self).predict_proba(X)

    def predict(self, X, **kwargs):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : The training input samples.  array-like or sparse matrix of shape = [n_samps, num_atts] or a data frame.
        If a Pandas data frame is passed, the column _dim_to_use is extracted
        Returns
        -------
        output : 1D array of predictions,
        """
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

        return super(FCN, self).score(X, y)



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

    nb_classes = np.unique(y_train).shape[0]
    batch_size = min(X_train.shape[0] // 10, 16)

    if len(X_train.iloc[0, 0].shape) == 1:
        input_shape = [X_train.iloc[0, 0].shape[0], 1]
    else:
        input_shape = X_train.iloc[0, 0].shape[0:]

    clf = FCN(nb_classes=nb_classes, batch_size=batch_size, input_shape=input_shape)

    hist = clf.fit(X_train, y_train)
    clf.model.summary()

    print(clf.score(X_test, y_test))

def test_sklearnPipeline():
    '''
    some sort of dependancy/versioning problem presumably, missing import deep within TSCTask, will sort during week if needed, else use above
    '''

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)

    # to be cleared up obviously, task/strategy want x/y combined, rest doesnt. correct way to handle?
    # need to just learn dataframes presumably
    train = load_gunpoint(split='TRAIN')
    test = load_gunpoint(split='TEST')

    nb_classes = np.unique(y_train).shape[0]
    batch_size = min(X_train.shape[0] // 10, 16)

    if len(X_train.iloc[0, 0].shape) == 1:
        input_shape = [X_train.iloc[0, 0].shape[0], 1]
    else:
        input_shape = X_train.iloc[0, 0].shape[0:]

    task = TSCTask(target='class_val', metadata=train)

    clf = FCN(nb_classes=nb_classes, batch_size=batch_size, input_shape=input_shape)
    strategy = TSCStrategy(clf)

    strategy.fit(task, train)

    y_pred = strategy.predict(test)
    #y_test = test[task.target]
    accuracy_score(y_test, y_pred)

if __name__ == "__main__":

    test_basic()            #working
    #test_sklearnPipeline() #broken, missing imports
