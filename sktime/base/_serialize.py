# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for serializing and deserializing objects."""

__author__ = ["fkiraly", "achieveordie"]


def load(serial):
    """Load object from serialized location - in-memory or file.

    Parameters
    ----------
    serial : serialized container, str (path), or ZipFile (reference)

    Returns
    -------
    deserialized self resulting in output `serial`, of `cls.save`

    Example
    -------
    >>> from sktime.base import load
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster()
    >>> forecaster.fit(y, fh=[1, 2, 3])
    NaiveForecaster()
    >>> pkl = forecaster.save()
    >>> forecaster_loaded = load(pkl)
    >>> y_pred = forecaster_loaded.predict()
    """
    from zipfile import ZipFile

    if isinstance(serial, tuple):
        cls = serial[0]
        stored = serial[1]
        return cls.load_from_serial(stored)
    elif isinstance(serial, (str, ZipFile)):

        if isinstance(serial, str):
            zipfile = ZipFile(serial)

        with zipfile.open("metadata", mode="r") as metadata:
            cls = metadata.read()
        with zipfile.open("object", mode="r") as object:
            return cls.load_from_path(object.read())
    else:
        raise TypeError(
            "serial must either be a serialized in-memory sktime object, "
            "or a str or ZipFile pointing to a file which is a serialized sktime "
            "object, created by save of an sktime object; but found serial "
            f"of type {serial}"
        )


def load_dl(serial):
    """Load Deep-Learning estimators from in-memory of saved folder.

    Parameters
    ----------
    serial : tuple or str or Path object
        if tuple then represents in-memory byte object of saved DL model
        if str / Path then points to the folder that was used with .save(path)

    Returns
    -------
    deserialized self resulting in output `serial`, of `cls.save`

    Example
    -------
    >>> from sktime.base import load_dl
    >>> import numpy as np
    >>> from sktime.classification.deep_learning import CNNClassifier
    >>> save_folder_location = "save_folder"
    >>> sample_X = np.random.randn(15, 24, 16)
    >>> sample_y = np.random.randint(0, 2, size=(15, ))
    >>> sample_test_X = np.random.randn(5, 24, 16)
    >>> cnn = CNNClassifier(n_epochs=1)
    >>> cnn.fit(sample_X, sample_y)
    CNNClassifier(n_epochs=1)
    >>> cnn.save(save_folder_location)
    >>> loaded_cnn = load_dl(save_folder_location)
    >>> pred = cnn.predict(X=sample_test_X)
    >>> loaded_pred = loaded_cnn.predict(X=sample_test_X)
    >>> print(np.allclose(pred, loaded_pred))
    True
    """
    import pickle
    from pathlib import Path

    if isinstance(serial, tuple):
        cls = serial[0]
        stored = serial[1]
        return cls.load_from_serial(stored)
    elif isinstance(serial, (str, Path)):
        if isinstance(serial, str):
            serial = Path(serial)

        if not serial.is_dir():
            raise TypeError(
                "`path` is expected to be a directory; but found "
                "`Path(serial).is_dir()` to be False"
            )
        if not serial.exists():
            raise FileNotFoundError(
                f"The given save directory: {serial}\nwas not found"
            )
        cls = pickle.load(open(serial / "_metadata", "rb"))
        return cls.load_from_path(serial)
    else:
        raise TypeError(
            "serial must either be a serialized in-memory sktime object, "
            "or a str or Path object pointing to a directory "
            "which is a serialized sktime object, created by save "
            f"of an sktime object; but found serial of type {serial}"
        )
