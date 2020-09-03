#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ForecastingHorizon"]

import numpy as np
import pandas as pd

from sktime.utils.validation import is_int
from sktime.utils.validation.forecasting import check_fh_values


class BaseForecastingHorizon:

    def __repr__(self):
        _repr = pd.Index.__repr__(self).strip(')')
        return f"{_repr}, is_relative={self.is_relative})"

    @staticmethod
    def _new(values, is_relative=True, **kwargs):
        return _make_fh(values, is_relative=is_relative)

    @staticmethod
    def _simple_new(values, is_relative=True, **kwargs):
        return _make_fh(values, is_relative=is_relative)

    def get_relative(self, cutoff=None):
        if self.is_relative:
            return self

        else:
            # dispatch on cutoff type
            self._check_cutoff(cutoff)
            values = self - cutoff

            if is_int(cutoff):
                pass

            # get integers
            elif isinstance(cutoff, pd.Period):
                values = [value.n for value in values]

            elif isinstance(cutoff, pd.Timestamp):
                try:
                    values = (values / pd.Timedelta(1, cutoff.freqstr)).astype(
                        np.int)
                except ValueError:
                    values = values.to_period(cutoff.freqstr) - pd.Period(
                        cutoff, freq=cutoff.freqstr)
                    values = [value.n for value in values]

            else:
                raise TypeError("`cutoff` type not supported")

            return self._new(values, is_relative=True)

    def get_absolute(self, cutoff=None):
        if not self.is_relative:
            return self

        else:
            self._check_cutoff(cutoff)

            if hasattr(cutoff, "freq"):
                # compute relative index for period index
                values = cutoff + self * cutoff.freq

            else:
                values = cutoff + self

            return self._new(values, is_relative=False)

    def get_in_sample(self, cutoff=None):
        relative = self.get_relative(cutoff)
        return self._new(self[relative <= 0],
                         is_relative=self.is_relative)

    def get_out_of_sample(self, cutoff=None):
        is_out_of_sample = self.get_relative(cutoff) > 0
        return self._new(self[is_out_of_sample],
                         is_relative=self.is_relative)

    def get_index_like(self, cutoff=None):
        return self.get_relative(cutoff=cutoff) - 1

    def _check_cutoff(self, cutoff):
        if cutoff is None:
            raise ValueError("`cutoff` must be provided.")

        if isinstance(self, pd.PeriodIndex):
            if not isinstance(cutoff, pd.Period):
                raise TypeError()

        elif isinstance(self, pd.DatetimeIndex):
            if not isinstance(cutoff, pd.Timestamp):
                raise TypeError()


def ForecastingHorizon(values, is_relative=True):
    """Factory method"""
    values = check_fh_values(values)
    return _make_fh(values, is_relative)


def _make_fh(values, is_relative):
    """Helper funtion to create classes for the forecasting horizons which
    dynaimcally inherit from pandas index types"""
    # get pandas index
    index = pd.Index(values)

    # define new class by dynamic inheritance from pandas index type
    class ForecastingHorizon(BaseForecastingHorizon, type(index)):
        """Forecasting Horizon"""

        def __new__(cls, *args, **kwargs):
            obj = object.__new__(cls)

            # update instances from index
            obj.__dict__.update(index.__dict__)
            obj.is_relative = is_relative

            return obj

    # create instance of new class
    return ForecastingHorizon(index, is_relative=is_relative)
