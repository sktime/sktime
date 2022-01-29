# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class templates for distances or kernels between time series, and for tabular data.

templates in this module:

    BasePairwiseTransformer - distances/kernels for tabular data
    BasePairwiseTransformerPanel - distances/kernels for time series

Interface specifications below.

---
    class name: BasePairwiseTransformer

Scitype defining methods:
    computing distance/kernel matrix (shorthand) - __call__(self, X, X2=X)
    computing distance/kernel matrix             - transform(self, X, X2=X)

Inspection methods:
    hyper-parameter inspection  - get_params()

---
    class name: BasePairwiseTransformerPanel

Scitype defining methods:
    computing distance/kernel matrix (shorthand) - __call__(self, X, X2=X)
    computing distance/kernel matrix             - transform(self, X, X2=X)

Inspection methods:
    hyper-parameter inspection  - get_params()
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.datatypes import check_is_scitype, convert_to
from sktime.datatypes._series_as_panel import convert_Series_to_Panel
from sktime.utils.validation.series import check_series


class BasePairwiseTransformer(BaseEstimator):
    """Base pairwise transformer for tabular or series data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "fit-in-transform": True,  # is "fit" empty? Yes, for all pairwise transforms
    }

    def __init__(self):
        super().__init__()
        self.X_equals_X2 = False

    def __call__(self, X, X2=None):
        """Compute distance/kernel matrix, call shorthand.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X

        alias for transform

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """
        # no input checks or input logic here, these are done in transform
        # this just defines __call__ as an alias for transform
        return self.transform(X=X, X2=X2)

    def transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """
        X = check_series(X)

        if X2 is None:
            X2 = X
            self.X_equals_X2 = True
        else:
            X2 = check_series(X2)

            def input_as_numpy(val):
                if isinstance(val, pd.DataFrame):
                    return val.to_numpy(copy=True)
                return val

            temp_X = input_as_numpy(X)
            temp_X2 = input_as_numpy(X2)
            if np.array_equal(temp_X, temp_X2):
                self.X_equals_X2 = True

        return self._transform(X=X, X2=X2)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from transform

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        raise NotImplementedError

    def fit(self, X=None, X2=None):
        """Fit method for interface compatibility (no logic inside)."""
        # no fitting logic, but in case fit is called or expected
        self._is_fitted = True
        return self


class BasePairwiseTransformerPanel(BaseEstimator):
    """Base pairwise transformer for panel data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "X_inner_mtype": "df-list",  # which mtype is used internally in _transform?
        "fit-in-transform": True,  # is "fit" empty? Yes, for all pairwise transforms
    }

    def __init__(self):
        super(BasePairwiseTransformerPanel, self).__init__()
        self.X_equals_X2 = False

    def __call__(self, X, X2=None):
        """Compute distance/kernel matrix, call shorthand.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X : Series or Panel, any supported mtype, of n instances
            Data to transform, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        X2 : Series or Panel, any supported mtype, of m instances
                optional, default: X = X2
            Data to transform, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
            X and X2 need not have the same mtype

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """
        # no input checks or input logic here, these are done in transform
        # this just defines __call__ as an alias for transform
        return self.transform(X=X, X2=X2)

    def transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X : Series or Panel, any supported mtype, of n instances
            Data to transform, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        X2 : Series or Panel, any supported mtype, of m instances
                optional, default: X = X2
            Data to transform, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
            X and X2 need not have the same mtype

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """
        X = self._pairwise_panel_x_check(X)

        if X2 is None:
            X2 = X
            self.X_equals_X2 = True
        else:
            X2 = self._pairwise_panel_x_check(X2, var_name="X2")
            # todo, possibly:
            # check X, X2 for equality, then set X_equals_X2
            # could use deep_equals

        return self._transform(X=X, X2=X2)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from transform

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X : guaranteed to be Series or Panel of mtype X_inner_mtype, n instances
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        X2 : guaranteed to be Series or Panel of mtype X_inner_mtype, m instances
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        raise NotImplementedError

    def fit(self, X=None, X2=None):
        """Fit method for interface compatibility (no logic inside)."""
        # no fitting logic, but in case fit is called or expected
        self._is_fitted = True
        return self

    def _pairwise_panel_x_check(self, X, var_name="X"):
        """Check and coerce input data.

        Method used to check the input and convert Series/Panel input
            to internally used format, as defined in X_inner_mtype tag

        Parameters
        ----------
        X: List of dfs, Numpy of dfs, 3d numpy
            The value to be checked
        var_name: str, variable name to print in error messages

        Returns
        -------
        X: Panel data container of a supported format in X_inner_mtype
            usually df-list, list of pd.DataFrame, unless overridden
        """
        check_res = check_is_scitype(
            X, ["Series", "Panel"], return_metadata=True, var_name=var_name
        )
        X_valid = check_res[0]
        metadata = check_res[2]

        X_scitype = metadata["scitype"]

        if not X_valid:
            raise TypeError("X/X2 must be of Series or Panel scitype")

        # if the input is a single series, convert it to a Panel
        if X_scitype == "Series":
            X = convert_Series_to_Panel(X)

        # can't be anything else if check_is_scitype is working properly
        elif X_scitype != "Panel":
            raise RuntimeError("Unexpected error in check_is_scitype, check validity")

        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_coerced = convert_to(X, to_type=X_inner_mtype, as_scitype="Panel")

        return X_coerced
