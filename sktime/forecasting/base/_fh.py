#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "ForecastingHorizon",
    "DELEGATED_METHODS"
]

import pandas as pd

from sktime.utils.time_series import _date_offsets_to_int
from sktime.utils.time_series import _timedeltas_to_int
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


def _delegator(method):
    # Helper function to automatically decorate ForecastingHorizon class
    # with methods from pandas.Index and delegate method calls to wrapped
    # pandas.Index object.
    def delegated(obj, *args, **kwargs):
        return getattr(obj.to_pandas(), method)(*args, **kwargs)

    return delegated


class ForecastingHorizon:
    """Forecasting horizon
    Parameters
    ----------
    values : pd.Index, np.array, list or int
        Values of forecasting horizon
    is_relative : bool, optional (default=True)
        - If True, values are relative to end of training series.
        - If False, values are absolute.
    """

    def __new__(cls, *args, **kwargs):
        # We want the ForecastingHorizon class to be an extension of a
        # pandas index, but since subclassing pandas indices is not
        # straightforward, we wrap the index object instead. In order to
        # still support the basic methods of a pandas index, we dynamically
        # add some basic methods and delegate the method calls to the wrapped
        # index object
        for method in DELEGATED_METHODS:
            setattr(cls, method, _delegator(method))
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

    def _new(self, values=None, is_relative=None):
        # convenience method for constructing new ForecastingHorizon based
        # on current object
        if values is None:
            values = self._values
        if is_relative is None:
            is_relative = self.is_relative
        return type(self)(values, is_relative)

    @property
    def is_relative(self):
        """Whether forecasting horizon represents relative values.

        Returns
        -------
        is_relative : bool
        """
        return self._is_relative

    def to_pandas(self):
        """Returns underlying values as pd.Index

        Returns
        -------
        fh : pd.Index
        """
        return self._values

    def to_numpy(self):
        """Returns underlying values as np.array

        Returns
        -------
        fh : np.ndarray
        """
        return self.to_pandas().to_numpy()

    def to_relative(self, cutoff=None):
        """Return relative values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            In-sample values of forecasting horizon
        """
        if self.is_relative:
            return self._new()

        else:
            self._check_cutoff(cutoff)
            values = self.to_pandas() - cutoff

            if isinstance(self.to_pandas(), pd.PeriodIndex):
                values = _date_offsets_to_int(values)

            if isinstance(self.to_pandas(), pd.DatetimeIndex):
                values = _timedeltas_to_int(values, cutoff.freqstr)

            return self._new(values, is_relative=True)

    def to_absolute(self, cutoff=None):
        """Return absolute values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            In-sample values of forecasting horizon
        """
        if not self.is_relative:
            return self._new()

        else:
            self._check_cutoff(cutoff)
            index = self.to_pandas()

            if hasattr(cutoff, "freq"):
                index *= cutoff.freq

            if isinstance(cutoff, pd.Timestamp):
                assert hasattr(cutoff, "freq")

            values = cutoff + index
            return self._new(values, is_relative=False)

    def to_in_sample(self, cutoff=None):
        """Return in-sample values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            In-sample values of forecasting horizon
        """
        is_in_sample = self.to_relative(cutoff).to_pandas() <= 0
        in_sample = self.to_pandas()[is_in_sample]
        return self._new(in_sample)

    def to_out_of_sample(self, cutoff=None):
        """Return out-of-sample values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            Out-of-sample values of forecasting horizon
        """
        is_out_of_sample = self.to_relative(cutoff).to_pandas() > 0
        out_of_sample = self.to_pandas()[is_out_of_sample]
        return self._new(out_of_sample)

    def to_indexer(self, cutoff=None, from_cutoff=True):
        """Return zero-based indexer values for easy indexing into arrays.
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.
        from_cutoff : bool, optional (default=True)
            - If True, zero-based relative to cutoff.
            - If False, zero-based relative to first value in forecasting
            horizon.

        Returns
        -------
        fh : pd.Index
            Indexer
        """
        if from_cutoff:
            return self.to_relative(cutoff).to_pandas() - 1
        else:
            relative = self.to_relative(cutoff)
            return relative - relative.to_pandas().min()

    def __repr__(self):
        class_name = self.__class__.__name__
        pandas_repr = repr(self.to_pandas()).split('(')[-1].strip(')')
        return f"{class_name}({pandas_repr}, is_relative={self.is_relative})"

    def __getitem__(self, item):
        # delegate getitem calls to wrapped pd.Index object
        return self.to_pandas()[item]

    def _check_cutoff(self, cutoff):
        """Helper method to check cutoff values"""
        if cutoff is None:
            raise ValueError("`cutoff` must be provided.")

        if isinstance(self.to_pandas(), pd.PeriodIndex):
            assert isinstance(cutoff, pd.Period)

        if isinstance(self.to_pandas(), pd.DatetimeIndex):
            assert isinstance(cutoff, pd.Timestamp)
