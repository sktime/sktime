#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["FH", "AbsoluteFH", "RelativeFH"]

import numpy as np
from sktime.utils.validation.forecasting import check_fh_values


class FH(np.ndarray):
    """Forecasting horizon"""

    is_relative = None
    _type = None

    def __new__(cls, values, is_relative=True):
        values = check_fh_values(values)
        if is_relative:
            klass = RelativeFH
        else:
            klass = AbsoluteFH
            if np.any(values < 0):
                raise ValueError("FH contains time points before observation horizon")
        return values.view(klass)

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


class RelativeFH(FH):
    is_relative = True
    _type = "relative"

    def relative(self, cutoff=None):
        return self

    def absolute(self, cutoff=None):
        self._check_cutoff(cutoff)
        return AbsoluteFH(self + cutoff)


class AbsoluteFH(FH):
    is_relative = False
    _type = "absolute"

    def relative(self, cutoff=None):
        self._check_cutoff(cutoff)
        return RelativeFH(self - cutoff)

    def absolute(self, cutoff=None):
        return self
