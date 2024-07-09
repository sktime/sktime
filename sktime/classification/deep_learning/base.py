"""Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""

__author__ = ["James-Large", "ABostrom", "TonyBagnall", "aurunmpegasus", "achieveordie"]
__all__ = ["BaseDeepClassifier"]

import os
from abc import abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

from sktime.base._base import SERIALIZATION_FORMATS
from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _check_soft_dependencies


class BaseDeepClassifier(BaseClassifier):
    """Abstract base class for deep learning time series classifiers.

    The base classifier provides a deep learning default method for
    _predict and _predict_proba, and provides a new abstract method for building a
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

    @abstractmethod
    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
            The number of classes, which shall become the size of the output
            layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def summary(self):
        """Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history: dict or None,
            Dictionary containing model's train/validation losses and metrics
        """
        return self.history.history if self.history is not None else None

    def _predict(self, X, **kwargs):
        probs = self._predict_proba(X, **kwargs)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def _predict_proba(self, X, **kwargs):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.         input_checks: boolean
            whether to check the X parameter

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        # Transpose to work correctly with keras
        X = X.transpose((0, 2, 1))
        probs = self.model_.predict(X, self.batch_size, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs

    def _convert_y_to_keras(self, y):
        """Convert y to required Keras format."""
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y = y.reshape(len(y), 1)

        # in sklearn 1.2, sparse was renamed to sparse_output
        if _check_soft_dependencies(
            "scikit-learn>=1.2",
            severity="none",
            package_import_alias={"scikit-learn": "sklearn"},
        ):
            sparse_kw = {"sparse_output": False}
        else:
            sparse_kw = {"sparse": False}

        self.onehot_encoder = OneHotEncoder(categories="auto", **sparse_kw)
        # categories='auto' to get rid of FutureWarning
        y = self.onehot_encoder.fit_transform(y)
        return y

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

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:
        if ``path`` is None, returns an in-memory serialized self
        if ``path`` is a file, stores the zip with that name at the location.
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
                path="estimator" then a zip file ``estimator.zip`` will be made at cwd.
                path="/home/stored/estimator" then a zip file ``estimator.zip`` will be
                stored in ``/home/stored/``.

        serialization_format : str, default = "pickle"
            Module to use for serialization.
            The available options are present under
            ``sktime.base._base.SERIALIZATION_FORMATS``. Note that non-default formats
            might require installation of other soft dependencies.

        Returns
        -------
        if ``path`` is None - in-memory serialized self
        if ``path`` is file location - ZipFile with reference to the file
        """
        import pickle
        from pathlib import Path

        if serialization_format not in SERIALIZATION_FORMATS:
            raise ValueError(
                f"The provided `serialization_format`='{serialization_format}' "
                "is not yet supported. The possible formats are: "
                f"{SERIALIZATION_FORMATS}."
            )

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type:{type(path)}."
            )

        if path is not None:
            path = Path(path) if isinstance(path, str) else path
            path.mkdir()

        if serialization_format == "cloudpickle":
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            serializer = cloudpickle
        elif serialization_format == "pickle":
            serializer = pickle

        return self._serialize_using_dump_func(
            path=path,
            dump=serializer.dump,
            dumps=serializer.dumps,
        )

    def _serialize_using_dump_func(self, path, dump, dumps):
        """Serialize & return DL Estimator using ``dump`` and ``dumps`` functions."""
        import shutil
        from zipfile import ZipFile

        history = self.history.history if self.history is not None else None
        if path is None:
            _check_soft_dependencies("h5py", severity="error")
            import h5py

            in_memory_model = None
            if self.model_ is not None:
                self.model_.save("disk_less.h5")
                with h5py.File("disk_less.h5", "r") as h5file:
                    in_memory_model = h5file.id.get_file_image()

            in_memory_history = dumps(history)
            return (
                type(self),
                (
                    dumps(self),
                    in_memory_model,
                    in_memory_history,
                ),
            )

        if self.model_ is not None:
            keras_path = path / "keras" / "model.keras"
            os.makedirs(keras_path.parent, exist_ok=True)
            self.model_.save(keras_path)

        with open(path / "history", "wb") as history_writer:
            dump(history, history_writer)
        with open(path / "_metadata", "wb") as file:
            dump(type(self), file)
        with open(path / "_obj", "wb") as file:
            dump(self, file)

        shutil.make_archive(base_name=path, format="zip", root_dir=path)
        shutil.rmtree(path)
        return ZipFile(path.with_name(f"{path.stem}.zip"))

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial: 1st element of output of ``cls.save(None)``
                This is a tuple of size 3.
                The first element represents pickle-serialized instance.
                The second element represents h5py-serialized ``keras`` model.
                The third element represent pickle-serialized history of ``.fit()``.

        Returns
        -------
        Deserialized self resulting in output ``serial``, of ``cls.save(None)``
        """
        _check_soft_dependencies("h5py")
        import pickle

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
            with open("diskless.h5", "wb") as store_:
                store_.write(in_memory_model)
                cls.model_ = load_model("diskless.h5")

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
        deserialized self resulting in output at ``path``, of ``cls.save(path)``
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

        keras_location_legacy = temp_unzip_loc / "keras"
        keras_location = temp_unzip_loc / "keras" / "model.keras"
        if keras_location.exists():
            cls.model_ = keras.models.load_model(keras_location)
        elif keras_location_legacy.exists():
            cls.model_ = keras.models.load_model(keras_location_legacy)
        else:
            cls.model_ = None

        rmtree(temp_unzip_loc)
        cls.history = keras.callbacks.History()
        with ZipFile(serial, mode="r") as file:
            cls.history.set_params(pickle.loads(file.open("history").read()))
            return pickle.loads(file.open("_obj").read())
