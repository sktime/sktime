"""Abstract base class for the Keras neural network regressors.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags and _predict
"""

__author__ = ["AurumnPegasus", "achieveordie"]
__all__ = ["BaseDeepRegressor"]

from abc import ABC, abstractmethod

import numpy as np

from sktime.regression.base import BaseRegressor
from sktime.utils.validation._dependencies import _check_soft_dependencies


class BaseDeepRegressor(BaseRegressor, ABC):
    """Abstract base class for deep learning time series regression.

    The base classifier provides a deep learning default method for
    _predict, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 40
        training batch size for the model

    Attributes
    ----------
    self.model_ - the fitted DL model
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(self, batch_size=40):
        super().__init__()

        self.batch_size = batch_size
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def _predict(self, X, **kwargs):
        """Find regression estimate for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.

        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        X = X.transpose((0, 2, 1))
        y_pred = self.model_.predict(X, self.batch_size, **kwargs)
        y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred

    def __getstate__(self):
        """Get Dict config that will be used when a serialization method is called.

        Returns
        -------
        copy : dict, the config to be serialized
        """
        from tensorflow.keras.optimizers import Optimizer, serialize

        copy = self.__dict__.copy()

        # Either optimizer might not exist at all(-1),
        # or it does and takes a value(including None)
        optimizer_attr = copy.get("optimizer", -1)
        if not isinstance(optimizer_attr, str):
            if optimizer_attr is None:
                # if it is None, then save it as 0, so it can be
                # later correctly restored as None
                copy["optimizer"] = 0
            elif optimizer_attr == -1:
                # if an `optimizer` parameter doesn't exist at all
                # save it as -1
                copy["optimizer"] = -1
            elif isinstance(optimizer_attr, Optimizer):
                copy["optimizer"] = serialize(optimizer_attr)
            else:
                raise ValueError(
                    f"`optimizer` of type {type(optimizer_attr)} cannot be "
                    "serialized, it should either be absent/None/str/"
                    "tf.keras.optimizers.Optimizer object"
                )
        else:
            # if it was a string, don't touch since already serializable
            pass

        check_before_deletion = ["model_", "history", "optimizer_"]
        for attribute in check_before_deletion:
            if copy.get(attribute) is not None:
                del copy[attribute]
        return copy

    def __setstate__(self, state):
        """Magic method called during deserialization.

        Parameters
        ----------
        state : dict, as returned from __getstate__(), used for correct deserialization

        Returns
        -------
        -
        """
        from tensorflow.keras.optimizers import deserialize

        self.__dict__ = state

        if hasattr(self, "model_"):
            self.__dict__["model_"] = self.model_
            if hasattr(self, "model_.optimizer"):
                self.__dict__["optimizer_"] = self.model_.optimizer

        # if optimizer_ exists, set optimizer as optimizer_
        if self.__dict__.get("optimizer_") is not None:
            self.__dict__["optimizer"] = self.__dict__["optimizer_"]
        # else model may not have been built, but an optimizer might be passed
        else:
            # Having 0 as value implies "optimizer" attribute was None
            # as per __getstate__()
            if self.__dict__.get("optimizer") == 0:
                self.__dict__["optimizer"] = None
            elif self.__dict__.get("optimizer") == -1:
                # `optimizer` doesn't exist as a parameter alone, so delete it.
                del self.__dict__["optimizer"]
            else:
                if isinstance(self.optimizer, dict):
                    self.__dict__["optimizer"] = deserialize(self.optimizer)
                else:
                    # must have been a string already, no need to set
                    pass

        if hasattr(self, "history"):
            self.__dict__["history"] = self.history

    def save(self, path=None):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:
        if `path` is None, returns an in-memory serialized self
        if `path` is a file, stores the zip with that name at the location.
        The contents of the zip file are:
        _metadata - contains class of self, i.e., type(self).
        _obj - serialized self. This class uses the default serialization (pickle).
        keras/ - model, optimizer and state stored inside this directory.
        history - serialized history object.


        Parameters
        ----------
        path : None or file location (str or Path)
            if None, self is saved to an in-memory object
            if file location, self is saved to that file location. For eg:
                path="estimator" then a zip file `estimator.zip` will be made at cwd.
                path="/home/stored/estimator" then a zip file `estimator.zip` will be
                stored in `/home/stored/`.

        Returns
        -------
        if `path` is None - in-memory serialized self
        if `path` is file location - ZipFile with reference to the file
        """
        import pickle
        import shutil
        from pathlib import Path
        from zipfile import ZipFile

        if path is None:
            _check_soft_dependencies("h5py")
            import h5py

            in_memory_model = None
            if self.model_ is not None:
                with h5py.File(
                    "disk_less", "w", driver="core", backing_store=False
                ) as h5file:
                    self.model_.save(h5file)
                    h5file.flush()
                    in_memory_model = h5file.id.get_file_image()

            in_memory_history = pickle.dumps(self.history.history)

            return (
                type(self),
                (
                    pickle.dumps(self),
                    in_memory_model,
                    in_memory_history,
                ),
            )

        if not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type:{type(path)}."
            )

        path = Path(path) if isinstance(path, str) else path
        path.mkdir()

        if self.model_ is not None:
            self.model_.save(path / "keras/")

        with open(path / "history", "wb") as history_writer:
            pickle.dump(self.history.history, history_writer)

        pickle.dump(type(self), open(path / "_metadata", "wb"))
        pickle.dump(self, open(path / "_obj", "wb"))

        shutil.make_archive(base_name=path, format="zip", root_dir=path)
        shutil.rmtree(path)
        return ZipFile(path.with_name(f"{path.stem}.zip"))

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial: 1st element of output of `cls.save(None)`
                This is a tuple of size 3.
                The first element represents pickle-serialized instance.
                The second element represents h5py-serialized `keras` model.
                The third element represent pickle-serialized history of `.fit()`.

        Returns
        -------
        Deserialized self resulting in output `serial`, of `cls.save(None)`
        """
        _check_soft_dependencies("h5py")
        import pickle
        from tempfile import TemporaryFile

        import h5py
        from tensorflow.keras.models import load_model

        if not isinstance(serial, tuple):
            raise TypeError(
                "`serial` is expected to be a tuple, "
                f"instead found of type: {type(serial)}"
            )
        if len(serial) != 3:
            raise ValueError(
                "`serial` should have 3 elements. "
                "All 3 elements represent in-memory serialization "
                "of the estimator. "
                f"Found a tuple of length: {len(serial)} instead."
            )

        serial, in_memory_model, in_memory_history = serial
        if in_memory_model is None:
            cls.model_ = None
        else:
            with TemporaryFile() as store_:
                store_.write(in_memory_model)
                h5file = h5py.File(store_, "r")
                cls.model_ = load_model(h5file)
                h5file.close()

        cls.history = pickle.loads(in_memory_history)
        return pickle.loads(serial)

    @classmethod
    def load_from_path(cls, serial):
        """Load object from file location.

        Parameters
        ----------
        serial : Name of the zip file.

        Returns
        -------
        deserialized self resulting in output at `path`, of `cls.save(path)`
        """
        import pickle
        from shutil import rmtree
        from zipfile import ZipFile

        from tensorflow import keras

        temp_unzip_loc = serial.parent / "temp_unzip/"
        temp_unzip_loc.mkdir()

        with ZipFile(serial, mode="r") as zip_file:
            for file in zip_file.namelist():
                if not file.startswith("keras/"):
                    continue
                zip_file.extract(file, temp_unzip_loc)

        keras_location = temp_unzip_loc / "keras"
        if keras_location.exists():
            cls.model_ = keras.models.load_model(keras_location)
        else:
            cls.model_ = None

        rmtree(temp_unzip_loc)
        cls.history = keras.callbacks.History()
        with ZipFile(serial, mode="r") as file:
            cls.history.set_params(pickle.loads(file.open("history").read()))
            return pickle.loads(file.open("_obj").read())
