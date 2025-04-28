"""Time Le-Net (TLENET)."""

__author__ = ["JamesLarge", "Withington"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_dl_dependencies


class TleNetNetwork(BaseDeepNetwork):
    """Time Le-Net (TLENET).

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/tlenet.py

    References
    ----------
    .. [1]  Network originally defined in:
    @inproceedings{wang2017time, title={Data augmentation for time series
    classification using convolutional neural networks}, author={Le Guennec,
    Arthur and Malinowski, Simon and Tavenard, Romain}, booktitle={ECML/PKDD
    workshop on advanced analytics and learning on temporal data},
    year={2016} }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(self, random_state=0):
        _check_dl_dependencies(severity="error")
        super().__init__()

        self.warping_ratios = [0.5, 1, 2]
        self.slice_ratio = 0.1
        self.random_state = random_state

    @staticmethod
    def slice_data(self, X, y=None, length_sliced=1):
        """
        Perform window slicing (WS) to provide additional data.

        "At training, each slice extracted from a time series of class y is
        assigned the same class and a classifier is learned using the slices.
        The size of the slice is a parameter of this method. At test time, each
        slice from a test time series is classified using the learned
        classifier and a majority vote is performed to decide a predicted
        label." Le Guennec et al. (2016)

        This method performs window slicing on input time series to
        generate additional data samples.

        Parameters
        ----------
        self : object
            The object instance.
        X : numpy.ndarray
            The input data with shape (samples, time_steps, features).
        y : numpy.ndarray or None, optional
            The target labels with shape (samples, labels), by default None.
        length_sliced : int, optional
            The length of each sliced window, by default 1.

        Returns
        -------
        new_x : numpy.ndarray
            The sliced input data with shape
                (augmented_samples, length_sliced, features).
        new_y : numpy.ndarray or None
            The replicated target labels with shape
                (augmented_samples, labels) if y is not None, else None.
        increase_num : int
            Number of augmented samples generated per original sample.

        """
        import numpy as np

        n = X.shape[0]
        length = X.shape[1]
        n_dim = X.shape[2]  # for MTS

        increase_num = (
            length - length_sliced + 1
        )  # if increase_num =5, it means one ori becomes 5 new instances.
        if increase_num < 0:
            raise Exception(
                "Number of augmented data samples cannot be \
                negative. Length of time series:",
                length,
                "Slice length:",
                length_sliced,
                ".",
            )
        n_sliced = n * increase_num

        # print((n_sliced, length_sliced, n_dim))

        new_x = np.zeros((n_sliced, length_sliced, n_dim))

        for i in range(n):
            for j in range(increase_num):
                new_x[i * increase_num + j, :, :] = X[i, j : j + length_sliced, :]

        # transform y, if present.
        new_y = None
        if y is not None:
            if len(y.shape) > 1:
                new_shape = (n_sliced, y.shape[1])
            else:
                new_shape = (n_sliced,)
            new_y = np.zeros(new_shape)
            for i in range(n):
                for j in range(increase_num):
                    new_y[i * increase_num + j] = y[i]

        return new_x, new_y, increase_num

    @staticmethod
    def window_warping(self, data_x, warping_ratio):
        """
        Warp a slice of a time series by speeding it up or down.

        "This method generates input time series of different lengths.
        To deal with this issue, we [then] perform window slicing on
        transformed timeseries for all to have equal length"
        Le Guennec et al. (2016)

        This method warps a slice of a time series by speeding it
        up or down using linear interpolation.

        Parameters
        ----------
        self : object
            The object instance.
        data_x : numpy.ndarray
            The input data with shape (samples, time_steps, features).
        warping_ratio : float
            The warping ratio determining the speed of the warping.

        Returns
        -------
        warped_series : numpy.ndarray
            The warped time series data with shape (samples, new_length, features).

        """
        import numpy as np

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

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        conv_1 = keras.layers.Conv1D(
            filters=5, kernel_size=5, activation="relu", padding="same"
        )(input_layer)
        conv_1 = keras.layers.MaxPool1D(pool_size=2)(conv_1)

        conv_2 = keras.layers.Conv1D(
            filters=20, kernel_size=5, activation="relu", padding="same"
        )(conv_1)
        conv_2 = keras.layers.MaxPool1D(pool_size=4)(conv_2)

        # they did not mention the number of hidden units in the
        # fully-connected layer so we took the lenet they referenced

        flatten_layer = keras.layers.Flatten()(conv_2)
        fully_connected_layer = keras.layers.Dense(500, activation="relu")(
            flatten_layer
        )

        return input_layer, fully_connected_layer

    @staticmethod
    def adjust_parameters(self, X):
        """Adjust warping and slicing parameters based on data length.

        This method adjusts warping and slicing parameters based on
        the length of input time series.

        Parameters
        ----------
        self : object
            The object instance.
        X : numpy.ndarray
            The input data with shape (samples, time_steps, features).
        """
        n = X.shape[0]  # num cases
        m = X.shape[1]  # series length

        # limit the number of augmented time series if series too long or
        # too many
        if m > 500 or n > 2000:
            self.warping_ratios = [1]
            self.slice_ratio = 0.9

        # Handle short series
        ratios = [i for i in self.warping_ratios if m / i >= 8]
        self.warping_ratios = ratios
        if m * self.slice_ratio < 8:
            self.slice_ratio = 8 / m  # increase the slice if series too short

    @staticmethod
    def pre_processing(self, X, y=None):
        """Perform data preprocessing including data augmentation.

        This method applies window warping (WW) and window slicing (WS)
        for data augmentation.

        Parameters
        ----------
        self : object
            The object instance.
        X : numpy.ndarray
            The input data with shape (samples, time_steps, features).
        y : numpy.ndarray or None, optional
            The target labels with shape (samples, labels), by default None.

        Returns
        -------
        new_x : numpy.ndarray
            The preprocessed input data after augmentation with shape
                (augmented_samples, length_ratio, features).
        new_y : numpy.ndarray or None
            The preprocessed target labels after augmentation with shape
                (augmented_samples, labels) if y is not None, else None.
        tot_increase_num : int
            Total number of augmented samples generated.
        """
        import numpy as np

        length_ratio = int(self.slice_ratio * X.shape[1])

        # list of the augmented as well as the original data
        x_augmented = []

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
            (
                x_augmented[i],
                y_train_augmented_i,
                increase_num,
            ) = self.slice_data(x_augmented[i], y, length_ratio)
            # print("inc num",increase_num)
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
                new_x[idx : idx + increase_num, :, :] = x_augmented[j][
                    i * increase_num : (i + 1) * increase_num, :, :
                ]
                idx += increase_num

        # merge y if its not None.
        new_y = None
        if y is not None:
            if len(y.shape) > 1:
                new_shape = (y.shape[0] * tot_increase_num, y.shape[1])
            else:
                new_shape = (y.shape[0] * tot_increase_num,)
            new_y = np.zeros(new_shape)
            idx = 0
            for i in range(X.shape[0]):
                for j in range(len(increase_nums)):
                    increase_num = increase_nums[j]
                    new_y[idx : idx + increase_num] = y_augmented[j][
                        i * increase_num : (i + 1) * increase_num
                    ]
                    idx += increase_num

        return new_x, new_y, tot_increase_num
