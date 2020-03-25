#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["BaseForecastingHorizon", "AbsoluteForecastingHorizon", "RelativeForecastingHorizon", "ForecastingHorizon"]

from sktime.utils.validation.forecasting import check_fh
import numpy as np


class BaseForecastingHorizon:
    is_relative = None
    _type = None

    def __init__(self, values):
        self._values = check_fh(values)

    def relative(self, cutoff=None):
        raise NotImplementedError("abstract method")

    def absolute(self, cutoff=None):
        raise NotImplementedError("abstract method")

    def _check_cutoff(self, cutoff):
        """Check cutoff"""
        if cutoff is None:
            output_type = "absolute" if self._type == "relative" else "relative"
            raise ValueError(f"When relative={self.is_relative}, the "
                             f"`cutoff` value must be passed in order to convert "
                             f"it to {output_type}, but found: None")

    def in_sample(self, cutoff=None):
        """Return in-sample values"""
        relative = self.relative(cutoff)
        return relative[relative <= 0]

    def out_of_sample(self, cutoff=None):
        """Return out-of-sample values"""
        relative = self.relative(cutoff)
        return relative[relative > 0]

    def index_like(self, cutoff=None):
        """Return zero-based index"""
        return self.relative(cutoff) - 1

    def __repr__(self):
        name = self.__class__.__name__
        values = repr(self._values)
        relative = self.is_relative
        return f"{name}(values={values}, relative={relative})"

    def __len__(self):
        return len(self._values)


class RelativeForecastingHorizon(BaseForecastingHorizon):
    is_relative = True
    _type = "relative"

    def relative(self, cutoff=None):
        return self._values

    def absolute(self, cutoff=None):
        self._check_cutoff(cutoff)
        values = self._values + cutoff
        if np.any(values < 0):
            raise ValueError("ForecastingHorizon contains time points before observation horizon")
        return AbsoluteForecastingHorizon(values)


class AbsoluteForecastingHorizon(BaseForecastingHorizon):
    is_relative = False
    _type = "absolute"

    def relative(self, cutoff=None):
        self._check_cutoff(cutoff)
        return self._values - cutoff

    def absolute(self, cutoff=None):
        return self._values


def ForecastingHorizon(values, relative=True):
    if relative:
        return RelativeForecastingHorizon(values)
    else:
        return AbsoluteForecastingHorizon(values)
