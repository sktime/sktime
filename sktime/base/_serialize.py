# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for serializing and deserializing objects.

IMPORTANT CAVEAT FOR DEVELOPERS:
Do not add estimator specific functionality to the `load` utility.
All estimator specific functionality should be in
the class methods `load_from_serial` and `load_from_path`.
"""

__author__ = ["fkiraly", "achieveordie"]


def load(serial):
    """Load an object either from in-memory object or from a file location.

    Parameters
    ----------
    serial : serialized container (tuple), str (path), or Path object (reference)
        if serial is a tuple (serialized container):
            Contains two elements, first in-memory metadata and second
            the related object.
        if serial is a string (path reference):
            The name of the file without the extension, for e.g: if the file
            is `estimator.zip`, `serial='estimator'`. It can also represent a
            path, for eg: if location is `home/stored/models/estimator.zip`
            then `serial='home/stored/models/estimator'`.
        if serial is a Path object (path reference):
            `serial` then points to the `.zip` file into which the
            object was stored using class method `.save()` of an estimator.

    Returns
    -------
    Deserialized self resulting in output `serial`, of `cls.save`

    Examples
    --------
    Example 1: saving an estimator in-memory and loading it back

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>>
    >>> # 1. fit the estimator
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster()
    >>> forecaster.fit(y, fh=[1, 2, 3])
    NaiveForecaster()
    >>>
    >>> # 2. save the fitted estimator
    >>> pkl = forecaster.save()
    >>>
    >>> # 3. load the saved estimator (can do this on empty kernel)
    >>> from sktime.base import load
    >>> forecaster_loaded = load(pkl)
    >>>
    >>> # 4. continue using the loaded estimator
    >>> y_pred = forecaster_loaded.predict()

    Example 2: saving a deep learning estimator on the hard drive and loading

    >>> import numpy as np
    >>> from sktime.classification.deep_learning import CNNClassifier
    >>>
    >>> # 1. fit the estimator
    >>> sample_X = np.random.randn(15, 24, 16) # doctest: +SKIP
    >>> sample_y = np.random.randint(0, 2, size=(15, )) # doctest: +SKIP
    >>> sample_test_X = np.random.randn(5, 24, 16) # doctest: +SKIP
    >>> cnn = CNNClassifier(n_epochs=1) # doctest: +SKIP
    >>> cnn.fit(sample_X, sample_y) # doctest: +SKIP
    >>>
    >>> # 2. save the fitted estimator
    >>> save_folder_location = "save_folder" # doctest: +SKIP
    >>> cnn.save(save_folder_location) # doctest: +SKIP
    >>>
    >>> # 3. load the saved estimator (can do this on empty kernel)
    >>> from sktime.base import load
    >>> save_folder_location = "save_folder" # doctest: +SKIP
    >>> loaded_cnn = load(save_folder_location) # doctest: +SKIP
    >>>
    >>> # 4. continue using the loaded estimator
    >>> pred = cnn.predict(X=sample_test_X) # doctest: +SKIP
    >>> loaded_pred = loaded_cnn.predict(X=sample_test_X) # doctest: +SKIP

    Example 3:  saving an estimator using cloudpickle's serialization functionality
                and loading it back
        Note: `cloudpickle` is a soft dependency and is not present
        with the base-installation.

    >>> from sktime.classification.feature_based import Catch22Classifier
    >>> from sktime.datasets import load_basic_motions  # doctest: +SKIP
    >>>
    >>> # 1. Fit the estimator
    >>> X_train, y_train = load_basic_motions(split="TRAIN")  # doctest: +SKIP
    >>> X_test, y_test = load_basic_motions(split="TEST")  # doctest: +SKIP
    >>> est = Catch22Classifier().fit(X_train, y_train)  # doctest: +SKIP
    >>>
    >>> # 2. save the fitted estimator
    >>> cpkl_serialized = est.save(serialization_format="cloudpickle")  # doctest: +SKIP
    >>>
    >>> # 3. load the saved estimator (possibly after sending it across a stream)
    >>> from sktime.base import load  # doctest: +SKIP
    >>> loaded_est = load(cpkl_serialized)  # doctest: +SKIP
    >>>
    >>> # 4. continue using the estimator as normal
    >>> pred = loaded_est.predict(X_test)    # doctest: +SKIP
    >>> loaded_pred = loaded_est.predict(X_test)  # doctest: +SKIP
    """
    import pickle
    from pathlib import Path
    from zipfile import ZipFile

    if isinstance(serial, tuple):
        if len(serial) != 2:
            raise ValueError(
                "`serial` should be a tuple of size 2 "
                f"found, a tuple of size: {len(serial)}"
            )
        cls, stored = serial
        return cls.load_from_serial(stored)

    elif isinstance(serial, (str, Path)):
        path = Path(serial + ".zip") if isinstance(serial, str) else serial
        if not path.exists():
            raise FileNotFoundError(f"The given save location: {serial}\nwas not found")
        with ZipFile(path) as file:
            cls = pickle.loads(file.open("_metadata", "r").read())
        return cls.load_from_path(path)
    else:
        raise TypeError(
            "serial must either be a serialized in-memory sktime object, "
            "a str, Path or ZipFile pointing to a file which is a serialized sktime "
            "object, created by save of an sktime object; but found serial "
            f"of type {serial}"
        )
