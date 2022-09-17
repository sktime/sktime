# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for serializing and deserializing objects."""

__author__ = ["fkiraly", "achieveordie"]


def load(serial):
    """Load an object either from in-memory or from a location.

    This location will be a file(for non-DL estimators) / folder(for DL estimators).

    Parameters
    ----------
    serial : tuple or string or ZipFile or Path object
        if serial is a tuple:
            Contains two elements, first in-memory metadata and second
            the related object.
        if serial is a string:
            Will be converted into a Path object, if resulting
            object is a file then calls internal classical loader
            if it is a folder then calls internal deep loader
            otherwise throws an error.
        if serial is a Zipfile:
            Calls to load internal classical loader.

    Returns
    -------
    Deserialized self resulting in output `serial`, of `cls.save`

    Examples
    --------
    >>> def load_classical_example():
    ...     from sktime.base import load
    ...     from sktime.datasets import load_airline
    ...     from sktime.forecasting.naive import NaiveForecaster
    ...     y = load_airline()
    ...     forecaster = NaiveForecaster()
    ...     forecaster.fit(y, fh=[1, 2, 3])
    ...     pkl = forecaster.save()
    ...     forecaster_loaded = load(pkl)
    ...     y_pred = forecaster_loaded.predict()
    >>> load_classical_example() # doctest: +SKIP
    >>> def load_dl_example():
    ...     from sktime.base import load
    ...     import numpy as np
    ...     from sktime.classification.deep_learning import CNNClassifier
    ...     save_folder_location = "save_folder"
    ...     sample_X = np.random.randn(15, 24, 16)
    ...     sample_y = np.random.randint(0, 2, size=(15, ))
    ...     sample_test_X = np.random.randn(5, 24, 16)
    ...     cnn = CNNClassifier(n_epochs=1)
    ...     cnn.fit(sample_X, sample_y)
    ...     cnn.save(save_folder_location)
    ...     loaded_cnn = load(save_folder_location)
    ...     pred = cnn.predict(X=sample_test_X)
    ...     loaded_pred = loaded_cnn.predict(X=sample_test_X)
    ...     print(np.allclose(pred, loaded_pred))
    ...     # True
    >>> load_dl_example() # doctest: +SKIP
    """
    from pathlib import Path
    from zipfile import ZipFile

    if isinstance(serial, tuple):
        cls = serial[0]
        stored = serial[1]
        return cls.load_from_serial(stored)

    elif isinstance(serial, (str, Path)):
        path = Path(serial) if isinstance(serial, str) else serial
        if not path.exists():
            raise FileNotFoundError(f"The given save location: {serial}\nwas not found")
        elif path.is_dir():
            return _load_dl(path)
        elif path.is_file():
            return _load_classical(path)
        else:
            raise TypeError("Shouldn't reach here, adding for completion")

    elif isinstance(serial, ZipFile):
        return _load_classical(serial)

    else:
        raise TypeError(
            "serial must either be a serialized in-memory sktime object, "
            "a str, Path or ZipFile pointing to a file which is a serialized sktime "
            "object, created by save of an sktime object; but found serial "
            f"of type {serial}"
        )


def _load_classical(serial):
    """Call functionality to load Non-DL estimators."""
    from pathlib import Path
    from zipfile import ZipFile

    zipfile = ZipFile(serial) if isinstance(serial, Path) else serial

    with zipfile.open("metadata", mode="r") as metadata:
        cls = metadata.read()
    with zipfile.open("object", mode="r") as object:
        return cls.load_from_path(object.read())


def _load_dl(serial):
    """Call functionality to load DL estimators."""
    import pickle

    cls = pickle.load(open(serial / "_metadata", "rb"))
    return cls.load_from_path(serial)
