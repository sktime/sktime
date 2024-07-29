"""Multi-scale Convolutional Neural Network Classifier."""

# todo: keras/tensorflow memory problem when search over network parameters
#      currently just deleting EVERY model and retraining the best parameters
#      at the end, see **1

import gc
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.mcnn import MCNNNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class MCNNClassifier(BaseDeepClassifier):
    """Multi-scale Convolutional Neural Network (MCNN), adapted from [1]_.

    Network is originally defined in [2]_.

    Parameters
    ----------
    pool_factors : list of int, optional (default=[2, 3, 5])
        The list of pooling factors, used for hyperparameter grid search.
    filter_size : list of float, optional (default=[0.05, 0.1, 0.2])
        The list of filter sizes, used for hyperparameter grid search.
    window_size : int, optional (default=3)
        The size of the window for the convolutional layer.
    nb_train_batch : int, optional (default=10)
        The number of training batches.
    nb_epochs : int, optional (default=200)
        The number of epochs to train the model.
    batch_size : int, optional (default=256)
        The maximum training batch size.
    slice_ratio : float, optional (default=0.9)
        The ratio to slice the training data.
    random_state : int, optional (default=0)
        The seed to any random action.
    verbose : bool, optional (default=False)
        Verbosity during model training, making it ``True`` will
        print model summary, training information etc.
    model_name : str, optional (default='mcnn')
        The name of the model.
    model_save_directory : str, optional (default=None)
        The directory to save the model.

    References
    ----------
    .. [1] Fawaz et. al https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcnn.py
    .. [2] Cui, Z., Chen, W., & Chen, Y. (2016). Multi-scale convolutional neural networks for time series classification. arXiv preprint arXiv:1603.06995.
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["fnhirwa"],
        "maintainers": "fnhirwa",
        "python_version": ">=3.9",
        "python_dependencies": "tensorflow==2.15.0",
        # estimator type handled by parent class
        "capability:predict": False,
    }

    def __init__(
        self,
        pool_factors=[2, 3, 5],
        filter_size=[0.05, 0.1, 0.2],
        window_size=3,
        nb_train_batch=10,
        nb_epochs=200,
        batch_size=256,
        slice_ratio=0.9,
        padding="same",
        random_state=0,
        verbose=False,
        callbacks=None,
        model_name="mcnn",
        model_save_directory=None,
    ):
        _check_dl_dependencies(severity="error")
        self.pool_factors = pool_factors
        self.filter_size = filter_size
        self.window_size = window_size
        self.nb_train_batch = nb_train_batch
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.slice_ratio = slice_ratio
        self.padding = padding
        self.random_state = random_state
        self.verbose = verbose
        self.callbacks = callbacks
        self.model_name = model_name
        self.model_save_directory = model_save_directory

        super().__init__()
        self._network = MCNNNetwork(
            pool_factor=2,
            kernel_size=7,
            padding="same",
            random_state=0,
        )
        self.input_shapes = None
        self.best_pool_factor = None
        self.best_filter_size = None
        self.history = None
        self._is_fitted = False

    def _set_hyperparameters(self):
        """Set up the ma and ds."""
        self.ma_base = 5
        self.ma_step = 6
        self.ma_num = 1
        self.ds_base = 2
        self.ds_step = 1
        self.ds_num = 4

    def _slice_data(
        self, X: np.ndarray, y: np.ndarray = None, slice_ratio: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target data.
        slice_ratio : float
            The ratio to slice the training data.

        Returns
        -------
        ret : tuple[np.ndarray, np.ndarray]
            A tuple of sliced data.
        """
        n = X.shape[0]
        length = X.shape[1]
        ndim = X.shape[2]  # for MTS
        sliced_length = int(length * slice_ratio)
        # if increase num = 5, means one ori becomes 5 new instances
        increase_num = length - sliced_length + 1

        n_sliced = n * increase_num
        sliced_X = np.zeros((n_sliced, sliced_length, ndim))

        sliced_y = None

        if y is not None:
            classes_num = y.shape[1]
            sliced_y = np.zeros((n_sliced, classes_num))

        for i in range(n):
            for j in range(increase_num):
                sliced_X[i * increase_num + j, :, :] = X[i, j : j + sliced_length, :]
                if y is not None:
                    sliced_y[i * increase_num + j] = np.int_(y[i].astype(np.float32))

        return sliced_X, sliced_y

    def _downsample(self, X, sample_rate, offset=0):
        """Downsample the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        sample_rate : int
            The sample rate.
        offset : int
            The offset.

        Returns
        -------
        ret : np.ndarray
            The downsampled data.
        """
        num = X.shape[0]
        X_length = X.shape[1]
        ndim = X.shape[2]  # for MTS
        last_one = 0
        if X_length % sample_rate > offset:
            last_one = 1
        new_length = int(np.floor(X_length / sample_rate)) + last_one
        res = np.zeros((num, new_length, ndim))

        for i in range(new_length):
            res[:, i] = np.array(X[:, offset + sample_rate * i])

        return res

    def _moving_average(self, X, window_size):
        """Compute the moving average.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        window_size : int
            The window size.

        Returns
        -------
        ret : np.ndarray
            The moving averaged data.
        """
        num = X.shape[0]
        length = X.shape[1]
        ndim = X.shape[2]  # for MTS
        new_length = length - window_size + 1
        res = np.zeros((num, new_length, ndim))

        for i in range(new_length):
            res[:, i] = np.mean(X[:, i : i + window_size], axis=1)
        return res

    def moving_average(self, X, window_base, step_size, num):
        """Compute the moving average.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        window_base : int
            The window base.
        step_size : int
            The step size.
        num : int
            The number of moving average.

        Returns
        -------
        ret : tuple[np.ndarray, list[int]]
            A tuple of moving averaged data and data lengths.
        """
        if num == 0:
            return (None, [])
        res = self._moving_average(X, window_base)
        data_lengths = [res.shape[1]]

        for i in range(1, num):
            window_size = window_base + step_size * i
            if window_size > X.shape[1]:
                continue
            new_series = self.moving_average(X, window_size)
            data_lengths.append(new_series.shape[1])
            res = np.concatenate([res, new_series], axis=1)
        return (res, data_lengths)

    def downsample(self, X, base, step_size, num):
        """Downsample the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        base : int
            The base.
        step_size : int
            The step size.
        num : int
            The number of downsampling.

        Returns
        -------
        ret : tuple[np.ndarray, list[int]]
            A tuple of downsampled data and data lengths.
        """
        # the case for dataset JapaneseVowels MTS
        if X.shape[1] == 26:
            return (None, [])  # too short to apply downsampling
        if num == 0:
            return (None, [])
        out = self._downsample(X, base, 0)
        data_lengths = [out.shape[1]]
        # for offset in range(1,base): #for the base case
        #    new_series = _downsample(X, base, offset)
        #    data_lengths.append( new_series.shape[1] )
        #    out = np.concatenate( [out, new_series], axis = 1)
        for i in range(1, num):
            sample_rate = base + step_size * i
            if sample_rate > X.shape[1]:
                continue
            for offset in range(0, 1):  # sample_rate):
                new_series = self._downsample(X, sample_rate, offset)
                data_lengths.append(new_series.shape[1])
                out = np.concatenate([out, new_series], axis=1)
        return (out, data_lengths)

    def build_model(self, input_shape, nb_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)
        n_classes: int
            The number of classes, which becomes the size of the output layer

        Returns
        -------
        output : a compiled Keras Model
        """
        if self._network is None:
            raise ValueError("The network has not been built yet.")
        import tensorflow as tf

        tf.random.set_seed(self.random_state)

        input_layer, fully_connected_layer = self._network.build_network(
            input_shape, **kwargs
        )

        output_layer = tf.keras.layers.Dense(
            units=nb_classes, activation="softmax", kernel_initializer="glorot_uniform"
        )(fully_connected_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=0.1
            ),  # runs slowly on M1/M2 Macs # noqa: E501
            metrics=["accuracy"],
        )

        return model

    def train(
        self,
        X,
        y,
        pool_factor,
        filter_size,
    ):
        """Train the model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target data.
        pool_factor : int
            The pool factor.
        filter_size : int
            The filter size.
        """
        # split train into validation set with validation_size = 0.2 train_size
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        ori_len = x_train.shape[1]  # original_length of time series

        kernel_size = int(ori_len * filter_size)

        # restrict slice ratio when data lenght is too large
        current_slice_ratio = self.slice_ratio
        if ori_len > 500:
            current_slice_ratio = self.slice_ratio if self.slice_ratio > 0.98 else 0.98

        increase_num = (
            ori_len - int(ori_len * current_slice_ratio) + 1
        )  # this can be used as the bath size
        # print(increase_num)

        train_batch_size = int(x_train.shape[0] * increase_num / self.nb_train_batch)
        current_n_train_batch = self.nb_train_batch
        if train_batch_size > self.batch_size:
            # limit the train_batch_size
            current_n_train_batch = int(
                x_train.shape[0] * increase_num / self.batch_size
            )

        # data augmentation by slicing the length of the series
        x_train, y_train = self._slice_data(x_train, y_train, current_slice_ratio)
        x_val, y_val = self._slice_data(x_val, y_val, current_slice_ratio)

        train_set_x, train_set_y = x_train, y_train
        valid_set_x, valid_set_y = x_val, y_val

        length_train = train_set_x.shape[1]  # length after slicing.

        current_window_size = (
            int(length_train * self.window_size)
            if self.window_size < 1
            else int(self.window_size)
        )

        ds_num_max = length_train / (pool_factor * current_window_size)
        current_ds_num = int(min(self.ds_num, ds_num_max))

        ma_train, ma_lengths = self.moving_average(
            train_set_x, self.ma_base, self.ma_step, self.ma_num
        )
        ma_valid, ma_lengths = self.moving_average(
            valid_set_x, self.ma_base, self.ma_step, self.ma_num
        )

        ds_train, ds_lengths = self.downsample(
            train_set_x, self.ds_base, self.ds_step, current_ds_num
        )
        ds_valid, ds_lengths = self.downsample(
            valid_set_x, self.ds_base, self.ds_step, current_ds_num
        )

        # concatenate directly
        data_lengths = [length_train]
        # downsample part:
        if ds_lengths != []:
            data_lengths += ds_lengths
            train_set_x = np.concatenate([train_set_x, ds_train], axis=1)
            valid_set_x = np.concatenate([valid_set_x, ds_valid], axis=1)

        # moving average part
        if ma_lengths != []:
            data_lengths += ma_lengths
            train_set_x = np.concatenate([train_set_x, ma_train], axis=1)
            valid_set_x = np.concatenate([valid_set_x, ma_valid], axis=1)
        # print("Data length:", data_lengths)

        n_train_size = train_set_x.shape[0]
        # n_valid_size = valid_set_x.shape[0]
        batch_size = int(n_train_size / current_n_train_batch)
        # data_dim = train_set_x.shape[1]
        num_dim = train_set_x.shape[2]  # For MTS
        nb_classes = train_set_y.shape[1]

        self.input_shapes, max_length = self.get_list_of_input_shapes(
            data_lengths, num_dim
        )
        # Create a new model for each set of hyperparameters
        self._network = MCNNNetwork(
            pool_factor=pool_factor,
            kernel_size=kernel_size,
            padding="same",
            random_state=self.random_state,
        )

        self.model_ = self.build_model(self.input_shapes, nb_classes)

        if self.verbose:
            self.model_.summary()
        self.callback_ = deepcopy(self.callbacks)

        x = self.split_input_for_model(train_set_x, self.input_shapes)
        x_val = self.split_input_for_model(valid_set_x, self.input_shapes)

        # Fit the model
        self.history = self.model_.fit(
            x,
            train_set_y,
            validation_data=(x_val, valid_set_y),
            batch_size=batch_size,
            epochs=self.nb_epochs,
            callbacks=self.callback_,
            verbose=self.verbose,
        )
        return self

    def split_input_for_model(self, x, input_shapes):
        """Split input for model compatibility.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        input_shapes : list of tuple
            The list of input shapes.

        Returns
        -------
        res : list of np.ndarray
            The list of input data.
        """
        res = []
        indx = 0
        for input_shape in input_shapes:
            res.append(x[:, indx : indx + input_shape[0], :])
            indx = indx + input_shape[0]
        return res

    def get_list_of_input_shapes(self, data_lengths, num_dim):
        """Get the list of input shapes.

        Parameters
        ----------
        data_lengths : list of int
            The list of data lengths.
        num_dim : int
            The number of dimensions.

        Returns
        -------
        input_shapes : list of tuple
            The list of input shapes.
        """
        input_shapes = []
        max_length = 0
        for i in data_lengths:
            input_shapes.append((i, num_dim))
            max_length = max(max_length, i)
        return input_shapes, max_length

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        y_onehot = self._convert_y_to_keras(y)
        X = X.transpose(0, 2, 1)
        check_random_state(self.random_state)
        self._set_hyperparameters()

        # best_df_metrics = None
        best_valid_loss = np.inf
        # grid search
        for pool_factor in self.pool_factors:
            for filter_size in self.filter_size:
                self.train(X, y_onehot, pool_factor, filter_size)
                valid_loss = self.history.history["val_loss"][-1]
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.best_pool_factor = pool_factor
                    self.best_filter_size = filter_size

                # clear memory in all the ways
                gc.collect()
                tf.keras.backend.clear_session()
        self.train(X, y_onehot, self.best_pool_factor, self.best_filter_size)
        self._is_fitted = True

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, optional (default="default")
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sktime.utils.dependencies import _check_soft_dependencies

        params = [
            {
                "pool_factors": [2, 3, 5],
                "filter_size": [0.05, 0.1, 0.2],
                "window_size": 3,
                "nb_train_batch": 10,
                "nb_epochs": 1,
                "batch_size": 256,
                "slice_ratio": 0.9,
                "random_state": 0,
                "verbose": False,
                "model_name": "mcnn",
                "model_save_directory": None,
            }
        ]
        if _check_soft_dependencies("keras", severity="none"):
            from keras.callbacks import EarlyStopping

            params.append(
                {
                    "pool_factors": [2, 3, 5],
                    "filter_size": [0.05, 0.1, 0.2],
                    "window_size": 3,
                    "nb_train_batch": 10,
                    "nb_epochs": 1,
                    "batch_size": 256,
                    "slice_ratio": 0.9,
                    "random_state": 0,
                    "verbose": False,
                    "callbacks": [EarlyStopping()],
                    "model_name": "mcnn",
                    "model_save_directory": None,
                }
            )
        return params
