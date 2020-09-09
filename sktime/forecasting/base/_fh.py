#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ForecastingHorizon"]

import numpy as np
import pandas as pd

from sktime.utils.validation.forecasting import check_fh_values

RELATIVE_TYPES = (
    pd.Int64Index,
    pd.RangeIndex
)
ABSOLUTE_TYPES = (
    pd.Int64Index,
    pd.RangeIndex,
    pd.DatetimeIndex,
    pd.PeriodIndex
)

DELEGATED_METHODS = (
    "__sub__",
    "__add__",
    "__mul__",
    "__div__",
    "__divmod__",
    "__pow__",
    "__gt__",
    "__ge__",
    "__ne__",
    "__lt__",
    "__eq__",
    "__le__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rdiv__",
    "__rmod__",
    "__rdivmod__",
    "__rpow__",
    "__len__"
)


def delegator(method):
    def delegated(obj, *args, **kwargs):
        return getattr(obj.to_pandas(), method)(*args, **kwargs)
    return delegated


class ForecastingHorizon:

    def __new__(cls, *args, **kwargs):
        # We want the ForecastingHorizon class to be an extension of a
        # pandas index, but since subclassing pandas indices is not
        # straightforward, we wrap the index object instead. In order to
        # still support the basic methods of a pandas index, we dynamically
        # add some basic methods and delegate the method calls to the wrapped
        # index object
        for method in DELEGATED_METHODS:
            setattr(cls, method, delegator(method))
        return object.__new__(cls)

    def __init__(self, values, is_relative=True):
        values = check_fh_values(values)
        assert isinstance(is_relative, bool)

        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we use check for type equality
        if is_relative:
            assert type(values) in RELATIVE_TYPES
        else:
            assert type(values) in ABSOLUTE_TYPES

        self._values = values
        self._is_relative = is_relative

    def _new(self, values, is_relative=None):
        if is_relative is None:
            is_relative = self.is_relative
        return type(self)(values, is_relative=is_relative)

    @property
    def is_relative(self):
        return self._is_relative

    def to_pandas(self):
        return self._values

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    def to_relative(self, cutoff=None):
        if self.is_relative:
            return self

        else:
            self._check_cutoff(cutoff)
            values = self.to_pandas() - cutoff

            if isinstance(self.to_pandas(), pd.PeriodIndex):
                values = date_offsets_to_int(values)

            if isinstance(self.to_pandas(), pd.DatetimeIndex):
                values = timedeltas_to_int(values, cutoff)

            return self._new(values, is_relative=True)

    def to_absolute(self, cutoff=None):
        if not self.is_relative:
            return self

        else:
            self._check_cutoff(cutoff)
            index = self.to_pandas()

            if hasattr(cutoff, "freq"):
                index *= cutoff.freq

            values = cutoff + index
            return self._new(values, is_relative=False)

    def to_in_sample(self, cutoff=None):
        is_in_sample = self.to_relative(cutoff).to_pandas() <= 0
        in_sample = self.to_pandas()[is_in_sample]
        return self._new(in_sample)

    def to_out_of_sample(self, cutoff=None):
        is_out_of_sample = self.to_relative(cutoff).to_pandas() > 0
        out_of_sample = self.to_pandas()[is_out_of_sample]
        return self._new(out_of_sample)

    def to_indexer(self, cutoff=None):
        return self.to_relative(cutoff).to_pandas() - 1

    def __repr__(self):
        class_name = self.__class__.__name__
        pandas_repr = repr(self.to_pandas()).split('(')[-1].strip(')')
        return f"{class_name}({pandas_repr}, is_relative={self.is_relative})"

    def __getitem__(self, item):
        return self.to_pandas()[item]

    def _check_cutoff(self, cutoff):
        if cutoff is None:
            raise ValueError("`cutoff` must be provided.")

        if isinstance(self.to_pandas(), pd.PeriodIndex):
            assert isinstance(cutoff, pd.Period)

        if isinstance(self.to_pandas(), pd.DatetimeIndex):
            assert isinstance(cutoff, pd.Timestamp)


def timedeltas_to_int(values, cutoff):
    assert type(values) == pd.TimedeltaIndex
    return (values / pd.Timedelta(1, cutoff.freqstr)).astype(np.int)


def date_offsets_to_int(values):
    assert type(values) == pd.Index
    assert isinstance(values[0], pd.tseries.offsets.DateOffset)
    return pd.Int64Index([value.n for value in values])
