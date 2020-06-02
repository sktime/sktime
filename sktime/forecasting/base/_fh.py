#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["FH", "AbsoluteFH", "RelativeFH"]

import numpy as np
import pandas as pd
from sktime.utils.validation.forecasting import check_fh_values


class FH(np.ndarray):
    """Forecasting horizon

    Parameters
    ----------
    values : np.array, list or int
        Values of forecasting horizon
    relative : bool, optional (default=True)
        - If True, values are relative to end of training series.
        - If False, values are absolute.
    """

    is_relative = None
    _type = None

    def __new__(cls, values, relative=True):
        """Construct forecasting horizon object"""

        if not relative and isinstance(values, pd.Index):
            values = values.values  # accept pandas index for absolute fh

        # input checks, returns numpy array
        values = check_fh_values(values)

        if relative:
            klass = RelativeFH
        else:
            klass = AbsoluteFH
            if np.any(values < 0):
                raise ValueError(
                    "FH contains time points before observation horizon")

        # subclass numpy array
        return values.view(klass)

    def relative(self, cutoff=None):
        raise NotImplementedError("abstract method")

    def absolute(self, cutoff=None):
        raise NotImplementedError("abstract method")

    def _check_cutoff(self, cutoff):
        """Check cutoff"""
        if cutoff is None:
            output_type = "absolute" if self._type == "relative" else \
                "relative"
            raise ValueError(f"When relative={self.is_relative}, the "
                             f"`cutoff` value must be passed in order to "
                             f"convert "
                             f"it to {output_type}, but found: None")

    def in_sample(self, cutoff=None):
        """Return in-sample values

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : in-sample values of forecasting horizon
        """
        relative = self.relative(cutoff)
        return relative[relative <= 0]

    def out_of_sample(self, cutoff=None):
        """Return out-of-sample values

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : out-of-sample values of forecasting horizon
        """
        relative = self.relative(cutoff)
        return relative[relative > 0]

    def index_like(self, cutoff=None):
        """Return zero-based index-like forecasting horizon.

        This is useful for indexing pd.Series using .iloc

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : relative zero-based index-like values of forecasting horizon
        """
        return self.relative(cutoff) - 1


class RelativeFH(FH):
    is_relative = True
    _type = "relative"

    def relative(self, cutoff=None):
        """Return forecasting horizon values relative to cutoff.

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : relative forecasting horizon
        """
        return self

    def absolute(self, cutoff=None):
        """Return absolute forecasting horizon values.

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : absolute forecasting horizon
        """
        self._check_cutoff(cutoff)
        values = self + cutoff
        return values.view(AbsoluteFH)


class AbsoluteFH(FH):
    is_relative = False
    _type = "absolute"

    def relative(self, cutoff=None):
        """Return forecasting horizon values relative to cutoff.

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : relative forecasting horizon
        """
        self._check_cutoff(cutoff)
        values = self - cutoff
        return values.view(RelativeFH)

    def absolute(self, cutoff=None):
        """Return absolute forecasting horizon values.

        Parameters
        ----------
        cutoff : int

        Returns
        -------
        fh : absolute forecasting horizon
        """
        return self
