# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for serializing and deserializing objects."""

__author__ = ["fkiraly"]


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
    >>> pkl = forecaster.save()
    >>> forecaster_loaded = load(pkl)
    >>> forecaster_loaded.predict()
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
