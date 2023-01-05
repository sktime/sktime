# -*- coding: utf-8 -*-

__author__ = ["James-Large", "Withington", "TonyBagnall",]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import (
    _check_dl_dependencies,
    _check_soft_dependencies,
)

_check_dl_dependencies(severity="warning")
_check_soft_dependencies(
    package_import_alias={"keras-self-attention": "keras_self_attention"},
    severity="warning",
)

_check_dl_dependencies(severity="warning")


class Classifier_MCNN:
    def __init__(self, output_directory, verbose, build=True):
        self.output_directory = output_directory
        self.verbose = verbose
        self.pool_factors = [2, 3, 5]  # used for hyperparameters grid search
        self.filter_sizes = [0.05, 0.1, 0.2]  # used for hyperparameters grid search

    def slice_data(self, data_x, data_y, slice_ratio):
        n = data_x.shape[0]
        length = data_x.shape[1]
        n_dim = data_x.shape[2]  # for MTS
        nb_classes = data_y.shape[1]

        length_sliced = int(length * slice_ratio)

        increase_num = (
            length - length_sliced + 1
        )  # if increase_num =5, it means one ori becomes 5 new instances.
        n_sliced = n * increase_num

        new_x = np.zeros((n_sliced, length_sliced, n_dim))
        new_y = np.zeros((n_sliced, nb_classes))
        for i in range(n):
            for j in range(increase_num):
                new_x[i * increase_num + j, :, :] = data_x[i, j : j + length_sliced, :]
                new_y[i * increase_num + j] = np.int_(data_y[i].astype(np.float32))

        return new_x, new_y

    def split_train(self, train_x, train_y):
        # shuffle for splitting train set and dataset
        n = train_x.shape[0]
        ind = np.arange(n)
        np.random.shuffle(ind)  # shuffle the train set

        # split train set into train set and validation set
        valid_x = train_x[ind[0 : int(0.2 * n)]]
        valid_y = train_y[ind[0 : int(0.2 * n)]]

        ind = np.delete(ind, (range(0, int(0.2 * n))))

        train_x = train_x[ind]
        train_y = train_y[ind]

        return train_x, train_y, valid_x, valid_y

    def _downsample(self, data_x, sample_rate, offset=0):
        num = data_x.shape[0]
        length_x = data_x.shape[1]
        num_dim = data_x.shape[2]  # for MTS
        last_one = 0
        if length_x % sample_rate > offset:
            last_one = 1
        new_length = int(np.floor(length_x / sample_rate)) + last_one
        output = np.zeros((num, new_length, num_dim))
        for i in range(new_length):
            output[:, i] = np.array(data_x[:, offset + sample_rate * i])

        return output

    def _movingavrg(self, data_x, window_size):
        num = data_x.shape[0]
        length_x = data_x.shape[1]
        num_dim = data_x.shape[2]  # for MTS
        output_len = length_x - window_size + 1
        output = np.zeros((num, output_len, num_dim))
        for i in range(output_len):
            output[:, i] = np.mean(data_x[:, i : i + window_size], axis=1)
        return output

    def movingavrg(self, data_x, window_base, step_size, num):
        if num == 0:
            return (None, [])
        out = self._movingavrg(data_x, window_base)
        data_lengths = [out.shape[1]]
        for i in range(1, num):
            window_size = window_base + step_size * i
            if window_size > data_x.shape[1]:
                continue
            new_series = self._movingavrg(data_x, window_size)
            data_lengths.append(new_series.shape[1])
            out = np.concatenate([out, new_series], axis=1)
        return (out, data_lengths)

    def batch_movingavrg(self, train, valid, test, window_base, step_size, num):
        (new_train, lengths) = self.movingavrg(train, window_base, step_size, num)
        (new_valid, lengths) = self.movingavrg(valid, window_base, step_size, num)
        (new_test, lengths) = self.movingavrg(test, window_base, step_size, num)
        return (new_train, new_valid, new_test, lengths)

    def downsample(self, data_x, base, step_size, num):
        # the case for dataset JapaneseVowels MTS
        if data_x.shape[1] == 26:
            return (None, [])  # too short to apply downsampling
        if num == 0:
            return (None, [])
        out = self._downsample(data_x, base, 0)
        data_lengths = [out.shape[1]]
        # for offset in range(1,base): #for the base case
        #    new_series = _downsample(data_x, base, offset)
        #    data_lengths.append( new_series.shape[1] )
        #    out = np.concatenate( [out, new_series], axis = 1)
        for i in range(1, num):
            sample_rate = base + step_size * i
            if sample_rate > data_x.shape[1]:
                continue
            for offset in range(0, 1):  # sample_rate):
                new_series = self._downsample(data_x, sample_rate, offset)
                data_lengths.append(new_series.shape[1])
                out = np.concatenate([out, new_series], axis=1)
        return (out, data_lengths)

    def batch_downsample(self, train, valid, test, window_base, step_size, num):
        (new_train, lengths) = self.downsample(train, window_base, step_size, num)
        (new_valid, lengths) = self.downsample(valid, window_base, step_size, num)
        (new_test, lengths) = self.downsample(test, window_base, step_size, num)
        return (new_train, new_valid, new_test, lengths)

    def get_pool_factor(self, conv_shape, pool_size):
        for pool_factor in self.pool_factors:
            temp_pool_size = int(int(conv_shape) / pool_factor)
            print(temp_pool_size)
            if temp_pool_size == pool_size:
                return pool_factor

        raise Exception("Error on pool factor")

    def train(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        y_true,
        pool_factor=None,
        filter_size=None,
        do_train=True,
    ):
        window_size = 0.2
        n_train_batch = 10
        n_epochs = 200
        max_train_batch_size = 256

        # print('Original train shape: ', x_train.shape)
        # print('Original test shape: ', x_test.shape)

        # split train into validation set with validation_size = 0.2 train_size
        x_train, y_train, x_val, y_val = self.split_train(x_train, y_train)

        ori_len = x_train.shape[1]  # original_length of time series
        slice_ratio = 0.9

        if do_train == True:
            kernel_size = int(ori_len * filter_size)

        if do_train == False:
            model = keras.models.load_model(self.output_directory + "best_model.hdf5")

            # model.summary()

            pool_size = model.get_layer("max_pooling1d_1").get_config()["pool_size"][0]

            conv_shape = model.get_layer("conv1d_1").output_shape[1]

            pool_factor = self.get_pool_factor(conv_shape, pool_size)

        # restrict slice ratio when data lenght is too large
        if ori_len > 500:
            slice_ratio = slice_ratio if slice_ratio > 0.98 else 0.98
        elif ori_len < 16:
            slice_ratio = 0.7

        increase_num = (
            ori_len - int(ori_len * slice_ratio) + 1
        )  # this can be used as the bath size

        train_batch_size = int(x_train.shape[0] * increase_num / n_train_batch)
        if train_batch_size > max_train_batch_size:
            # limit the train_batch_size
            n_train_batch = int(x_train.shape[0] * increase_num / max_train_batch_size)

        # data augmentation by slicing the length of the series
        x_train, y_train = self.slice_data(x_train, y_train, slice_ratio)
        x_val, y_val = self.slice_data(x_val, y_val, slice_ratio)
        x_test, y_test = self.slice_data(x_test, y_test, slice_ratio)

        train_set_x, train_set_y = x_train, y_train
        valid_set_x, valid_set_y = x_val, y_val
        test_set_x, _ = x_test, y_test

        valid_num = valid_set_x.shape[0]

        # print("increase factor is ", increase_num, ', ori len', ori_len)
        valid_num_batch = int(valid_num / increase_num)

        test_num = test_set_x.shape[0]
        test_num_batch = int(test_num / increase_num)

        length_train = train_set_x.shape[1]  # length after slicing.

        window_size = (
            int(length_train * window_size) if window_size < 1 else int(window_size)
        )

        # *******set up the ma and ds********#
        ma_base, ma_step, ma_num = 5, 6, 1
        ds_base, ds_step, ds_num = 2, 1, 4

        ds_num_max = length_train / (pool_factor * window_size)
        ds_num = int(min(ds_num, ds_num_max))

        # *******set up the ma and ds********#

        (ma_train, ma_valid, ma_test, ma_lengths) = self.batch_movingavrg(
            train_set_x, valid_set_x, test_set_x, ma_base, ma_step, ma_num
        )
        (ds_train, ds_valid, ds_test, ds_lengths) = self.batch_downsample(
            train_set_x, valid_set_x, test_set_x, ds_base, ds_step, ds_num
        )

        # concatenate directly
        data_lengths = [length_train]
        # downsample part:
        if ds_lengths != []:
            data_lengths += ds_lengths
            train_set_x = np.concatenate([train_set_x, ds_train], axis=1)
            valid_set_x = np.concatenate([valid_set_x, ds_valid], axis=1)
            test_set_x = np.concatenate([test_set_x, ds_test], axis=1)

        # moving average part
        if ma_lengths != []:
            data_lengths += ma_lengths
            train_set_x = np.concatenate([train_set_x, ma_train], axis=1)
            valid_set_x = np.concatenate([valid_set_x, ma_valid], axis=1)
            test_set_x = np.concatenate([test_set_x, ma_test], axis=1)
        # print("Data length:", data_lengths)

        n_train_size = train_set_x.shape[0]
        n_valid_size = valid_set_x.shape[0]
        n_test_size = test_set_x.shape[0]
        batch_size = int(n_train_size / n_train_batch)
        n_train_batches = int(n_train_size / batch_size)
        data_dim = train_set_x.shape[1]
        num_dim = train_set_x.shape[2]  # For MTS
        nb_classes = train_set_y.shape[1]

        # print('train size', n_train_size, ',valid size', n_valid_size, ' test size', n_test_size)
        # print('batch size ', batch_size)
        # print('n_train_batches is ', n_train_batches)
        # print('data dim is ', data_dim)
        # print('---------------------------')

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        # print('building the model...')

        input_shapes, max_length = self.get_list_of_input_shapes(data_lengths, num_dim)

        start_time = time.time()

        best_validation_loss = np.inf

        if do_train == True:

            model = self.build_model(input_shapes, nb_classes, pool_factor, kernel_size)

            if self.verbose == True:
                model.summary()

            # print('Training')

            # early-stopping parameters
            patience = 10000  # look as this many examples regardless
            patience_increase = 2  # wait this much longer when a new best is
            # found
            improvement_threshold = 0.995  # a relative improvement of this much is
            # considered significant
            validation_frequency = min(n_train_batches, patience / 2)
            # go through this many
            # minibatche before checking the network
            # on the validation set; in this case we
            # check every epoch
            max_before_stopping = 500

            best_iter = 0
            valid_loss = 0.0

            epoch = 0
            done_looping = False
            num_no_update_epoch = 0
            epoch_avg_cost = float("inf")
            epoch_avg_err = float("inf")

            while (epoch < n_epochs) and (not done_looping):
                epoch = epoch + 1
                epoch_train_err = 0.0
                epoch_cost = 0.0

                num_no_update_epoch += 1
                if num_no_update_epoch == max_before_stopping:
                    break

                for minibatch_index in range(n_train_batches):

                    iteration = (epoch - 1) * n_train_batches + minibatch_index

                    x = train_set_x[
                        minibatch_index
                        * batch_size : (minibatch_index + 1)
                        * batch_size
                    ]
                    y = train_set_y[
                        minibatch_index
                        * batch_size : (minibatch_index + 1)
                        * batch_size
                    ]

                    x = self.split_input_for_model(x, input_shapes)

                    cost_ij, accuracy = model.train_on_batch(x, y)

                    train_err = 1 - accuracy

                    epoch_train_err = epoch_train_err + train_err
                    epoch_cost = epoch_cost + cost_ij

                    if (iteration + 1) % validation_frequency == 0:

                        valid_losses = []
                        for i in range(valid_num_batch):
                            x = valid_set_x[
                                i * (increase_num) : (i + 1) * (increase_num)
                            ]
                            y_pred = model.predict_on_batch(
                                self.split_input_for_model(x, input_shapes)
                            )

                            # convert the predicted from binary to integer
                            y_pred = np.argmax(y_pred, axis=1)
                            label = np.argmax(valid_set_y[i * increase_num])

                            unique_value, sub_ind, correspond_ind, count = np.unique(
                                y_pred, True, True, True
                            )
                            unique_value = unique_value.tolist()

                            curr_err = 1.0
                            if label in unique_value:
                                target_ind = unique_value.index(label)
                                count = count.tolist()
                                sorted_count = sorted(count)
                                if count[target_ind] == sorted_count[-1]:
                                    if (
                                        len(sorted_count) > 1
                                        and sorted_count[-1] == sorted_count[-2]
                                    ):
                                        curr_err = 0.5  # tie
                                    else:
                                        curr_err = 0
                            valid_losses.append(curr_err)
                        valid_loss = sum(valid_losses) / float(len(valid_losses))

                        # print('...epoch%i,valid err: %.5f |' % (epoch,valid_loss))

                        # if we got the best validation score until now
                        if valid_loss <= best_validation_loss:
                            num_no_update_epoch = 0

                            # improve patience if loss improvement is good enough
                            if (
                                valid_loss
                                < best_validation_loss * improvement_threshold
                            ):
                                patience = max(patience, iteration * patience_increase)

                            # save best validation score and iteration number
                            best_validation_loss = valid_loss
                            best_iter = iteration

                            # save model in h5 format
                            model.save(self.output_directory + "best_model.hdf5")

                        model.save(self.output_directory + "last_model.hdf5")
                    if patience <= iteration:
                        done_looping = True
                        break
                epoch_avg_cost = epoch_cost / n_train_batches
                epoch_avg_err = epoch_train_err / n_train_batches

                # print ('train err %.5f, cost %.4f' %(epoch_avg_err,epoch_avg_cost))
                if epoch_avg_cost == 0:
                    break

            # print('Optimization complete.')

        # test the model
        # print('Testing')
        # load best model
        model = keras.models.load_model(self.output_directory + "best_model.hdf5")

        # get the true predictions of the test set
        y_predicted = []
        for i in range(test_num_batch):
            x = test_set_x[i * (increase_num) : (i + 1) * (increase_num)]
            y_pred = model.predict_on_batch(self.split_input_for_model(x, input_shapes))

            # convert the predicted from binary to integer
            y_pred = np.argmax(y_pred, axis=1)

            unique_value, sub_ind, correspond_ind, count = np.unique(
                y_pred, True, True, True
            )

            idx_max = np.argmax(count)
            predicted_label = unique_value[idx_max]

            y_predicted.append(predicted_label)

        y_pred = np.array(y_predicted)

        duration = time.time() - start_time

        df_metrics = calculate_metrics(y_true, y_pred, duration)

        # print(y_true.shape)
        # print(y_pred.shape)

        df_metrics.to_csv(self.output_directory + "df_metrics.csv", index=False)

        return df_metrics, model, best_validation_loss

    def split_input_for_model(self, x, input_shapes):
        res = []
        indx = 0
        for input_shape in input_shapes:
            res.append(x[:, indx : indx + input_shape[0], :])
            indx = indx + input_shape[0]
        return res

    def build_model(self, input_shapes, nb_classes, pool_factor, kernel_size):
        input_layers = []
        stage_1_layers = []

        for input_shape in input_shapes:

            input_layer = keras.layers.Input(input_shape)

            input_layers.append(input_layer)

            conv_layer = keras.layers.Conv1D(
                filters=256,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                kernel_initializer="glorot_uniform",
            )(input_layer)

            # should all concatenated have the same length
            pool_size = int(int(conv_layer.shape[1]) / pool_factor)

            max_layer = keras.layers.MaxPooling1D(pool_size=pool_size)(conv_layer)

            # max_layer = keras.layers.GlobalMaxPooling1D()(conv_layer)

            stage_1_layers.append(max_layer)

        concat_layer = keras.layers.Concatenate(axis=-1)(stage_1_layers)

        kernel_size = int(
            min(kernel_size, int(concat_layer.shape[1]))
        )  # kernel shouldn't exceed the length

        full_conv = keras.layers.Conv1D(
            filters=256,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
        )(concat_layer)

        pool_size = int(int(full_conv.shape[1]) / pool_factor)

        full_max = keras.layers.MaxPooling1D(pool_size=pool_size)(full_conv)

        full_max = keras.layers.Flatten()(full_max)

        fully_connected = keras.layers.Dense(
            units=256, activation="sigmoid", kernel_initializer="glorot_uniform"
        )(full_max)

        output_layer = keras.layers.Dense(
            units=nb_classes, activation="softmax", kernel_initializer="glorot_uniform"
        )(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.1),
            metrics=["accuracy"],
        )

        return model

    def get_list_of_input_shapes(self, data_lengths, num_dim):
        input_shapes = []
        max_length = 0
        for i in data_lengths:
            input_shapes.append((i, num_dim))
            max_length = max(max_length, i)
        return input_shapes, max_length

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        if not tf.test.is_gpu_available:
            print("error")
            exit()
        best_df_metrics = None
        best_valid_loss = np.inf

        output_directory_root = self.output_directory
        # grid search
        for pool_factor in self.pool_factors:
            for filter_size in self.filter_sizes:
                self.output_directory = (
                    output_directory_root
                    + "/hyper_param_search/"
                    + "/pool_factor_"
                    + str(pool_factor)
                    + "/filter_size_"
                    + str(filter_size)
                    + "/"
                )
                create_directory(self.output_directory)
                df_metrics, model, valid_loss = self.train(
                    x_train, y_train, x_test, y_test, y_true, pool_factor, filter_size
                )

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_df_metrics = df_metrics
                    best_df_metrics.to_csv(
                        output_directory_root + "df_metrics.csv", index=False
                    )
                    model.save(output_directory_root + "best_model.hdf5")

                model = None
                # clear memeory
                keras.backend.clear_session()

    def predict(self, x_test, y_true, x_train, y_train, y_test):
        df_metrics, _, _ = self.train(
            x_train, y_train, x_test, y_test, y_true, do_train=False
        )

        return df_metrics
