# -*- coding: utf-8 -*-
"""
Base class template for transformers.

    class name: BaseTransformer

Covers all types of transformers.
Type and behaviour of transformer is determined by the following tags:
    "scitype:transform-output" tag with values "Primitives", or "Series"
        this determines type of output of transform
        if "Primitives", output is pd.DataFrame with as many rows as X has instances
            i-th instance of X is transformed into i-th row of output
        if "Series", output is a Series or Panel, with as many instances as X
            i-th instance of X is transformed into i-th instance of output
        Series are treated as one-instance-Panels
            if Series is input, output is a 1-row pd.DataFrame or a Series
    "scitype:instancewise" tag which is boolean
        if True, fit/transform is statistically independent by instance

Scitype defining methods:
    fitting         - fit(self, X, y=None)
    transform       - transform(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)
    updating        - update(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["mloning, fkiraly"]
__all__ = [
    "BaseTransformer",
    "_SeriesToPrimitivesTransformer",
    "_SeriesToSeriesTransformer",
    "_PanelToTabularTransformer",
    "_PanelToPanelTransformer",
]

import warnings

from typing import Union

import numpy as np
import pandas as pd

from sklearn.base import clone

from sktime.base import BaseEstimator
from sktime.datatypes import convert_to, mtype, mtype_to_scitype, check_is
from sktime.datatypes._series_as_panel import (
    convert_Series_to_Panel, convert_Panel_to_Series
)


# single/multiple primitives
Primitive = Union[np.integer, int, np.float, float, str]
Primitives = np.ndarray

# tabular/cross-sectional data
Tabular = Union[pd.DataFrame, np.ndarray]  # 2d arrays

# univariate/multivariate series
UnivariateSeries = Union[pd.Series, np.ndarray]
MultivariateSeries = Union[pd.DataFrame, np.ndarray]
Series = Union[UnivariateSeries, MultivariateSeries]

# panel/longitudinal/series-as-features data
Panel = Union[pd.DataFrame, np.ndarray]  # 3d or nested array


class BaseTransformer(BaseEstimator):
    """Transformer base class."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
        "fit-in-transform": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
    }

    def __init__(self):

        self._is_fitted = False
        super(BaseTransformer, self).__init__()

    def fit(self, X, y=None, Z=None):
        """Fit transformer to X, optionally to y.

        By default, fit is empty. Fittable transformations overwrite fit method.

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to fit transform to
        y : Series or Panel, optional (default=None)
            Additional data, e.g., labels for transformation
        Z : possible alias for X; should not be passed when X is passed
            alias Z will be deprecated in version 0.9.0

        Returns
        -------
        self : a fitted instance of the estimator
        """
        X = _handle_alias(X, Z)

        self._is_fitted = False

        # skip everything if fit-in-transform is True
        if self.get_tag("fit-in-transform"):
            self._is_fitted = True
            return self

        # retrieve mtypes/scitypes of all objects
        #########################################

        X_mtype = mtype(X)
        X_scitype = mtype_to_scitype(X_mtype)
        y_mtype = mtype(y)
        y_scitype = mtype_to_scitype(y_mtype)

        X_inner_mtype = self.get_tag("X_inner_mtype")
        if not isinstance(X_inner_mtype, list):
            X_inner_mtype = [X_inner_mtype]
        X_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in X_inner_mtype]))

        y_inner_mtype = self.get_tag("y_inner_mtype")
        if not isinstance(X_inner_mtype, list):
            y_inner_mtype = [y_inner_mtype]
        y_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in y_inner_mtype]))

        # treating Series vs Panel conversion for X
        ###########################################

        # there are three cases to treat:
        # 1. if the internal _fit supports X's scitype, move on to mtype conversion
        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        # 3. internal only has Series but X is Panel:  loop over instances
        #     currently this is enabled by conversion to df-list mtype
        #     and this does not support y (unclear what should happen here)

        # 1. nothing to do - simply don't enter any of the ifs below

        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        if X_scitype == "Series" and "Series" not in X_inner_scitypes:
            X = convert_Series_to_Panel(X)
            X_mtype = mtype(X)
            X_scitype = mtype_to_scitype(X_mtype)

        # 3. internal only has Series but X is Panel: loop over instances
        if X_scitype == "Panel" and "Panel" not in X_inner_scitypes:
            if y is not None:
                raise ValueError(
                    "no default behaviour if _fit does not support Panel, "
                    " but X is Panel and y is not None"
                )
            X = convert_to(X, to_type="df-list", as_scitype="Panel")
            # this fits one transformer per instance
            self.transformers_ = [clone(self).fit(Xi) for Xi in X]
            # recurse and leave function - recursion does input checks/conversion
            return self

        # input checks and minor coercions on X, y
        ###########################################

        valid, msg, metadata = check_is(
            X, mtype=X_mtype, return_metadata=True, var_name="X"
        )
        if not valid:
            raise ValueError(msg)

        # checking X
        enforce_univariate = self.get_tag("univariate_only")
        if enforce_univariate and not metadata["is_univariate"]:
            raise ValueError("X must be univariate but is not")

        # convert X/y to supported inner type, if necessary
        ###################################################

        # subset to the mtypes that are of the same scitype as X/y
        X_inner_mtype = [
            mt for mt in X_inner_mtype if mtype_to_scitype(mt) == X_scitype
        ]

        y_inner_mtype = [
            mt for mt in y_inner_mtype if mtype_to_scitype(mt) == y_scitype
        ]

        # convert X and y to a supported internal type
        #  if X/y type is already supported, no conversion takes place
        X_inner = convert_to(
            X,
            to_type=X_inner_mtype,
            as_scitype=X_scitype,
        )
        y_inner = convert_to(
            y,
            to_type=y_inner_mtype,
            as_scitype=y_scitype,
        )

        self._fit(X=X_inner, y=y_inner)

        self._is_fitted = True
        return self

    def transform(self, X, y=None, Z=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed
        y : Series or Panel, optional (default=None)
            Additional data, e.g., labels for transformation
        Z : possible alias for X; should not be passed when X is passed
            alias Z will be deprecated in version 0.9.0

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported
        """
        X = _handle_alias(X, Z)

        # check whether is fitted, unless fit-in-transform is true
        if self.get_tag("fit-in-transform"):
            self._is_fitted = True
        elif not self._is_fitted:
            raise RuntimeError("fit must be called before transform")

        # retrieve mtypes/scitypes of all objects
        #########################################

        X_mtype = mtype(X)
        X_scitype = mtype_to_scitype(X_mtype)
        y_mtype = mtype(y)
        y_scitype = mtype_to_scitype(y_mtype)

        output_scitype = self.get_tag("scitype:transform-output")

        X_inner_mtype = self.get_tag("X_inner_mtype")
        if not isinstance(X_inner_mtype, list):
            X_inner_mtype = [X_inner_mtype]
        X_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in X_inner_mtype]))

        y_inner_mtype = self.get_tag("y_inner_mtype")
        if not isinstance(X_inner_mtype, list):
            y_inner_mtype = [y_inner_mtype]
        y_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in y_inner_mtype]))

        # treating Series vs Panel conversion for X
        ###########################################

        # there are three cases to treat:
        # 1. if the internal _fit supports X's scitype, move on to mtype conversion
        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        # 3. internal only has Series but X is Panel:  loop over instances
        #     currently this is enabled by conversion to df-list mtype
        #     and this does not support y (unclear what should happen here)

        # 1. nothing to do - simply don't enter any of the ifs below

        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        if X_scitype == "Series" and "Series" not in X_inner_scitypes:
            X_orig_mtype = X_mtype
            X_orig_scitype = X_scitype
            X = convert_Series_to_Panel(X)
            X_mtype = mtype(X)
            X_scitype = mtype_to_scitype(X_mtype)
            # remember that we did this
            X_was_Series = True
        else:
            X_was_Series = False
            X_orig_mtype = X_mtype
            X_orig_scitype = X_scitype

        # 3. internal only has Series but X is Panel: loop over instances
        if X_scitype == "Panel" and "Panel" not in X_inner_scitypes:
            if y is not None:
                ValueError(
                    "no default behaviour if _fit does not support Panel, "
                    " but X is Panel and y is not None"
                )
            X = convert_to(X, to_type="df-list", as_scitype="Panel")

            if self.get_tag("fit-in-transform"):
                Xt = [clone(self).transform(Xi) for Xi in X]
            else:
                fitted_trafos = self.transformers_
                if len(fitted_trafos) != len(X):
                    raise RuntimeError(
                        "found different number of instances in transform than in fit"
                    )
                else:
                    Xt = [fitted_trafos[i].transform(X[i]) for i in range(len(X))]
            # now we have a list of transformed instances

            # if the output is Series, Xt is a Panel and we convert back
            if output_scitype == "Series":
                Xt = convert_to(X, to_type=X_mtype, as_scitype="Panel")
            # if the output is Primitives, we have a list of one-row dataframes
            # we concatenate those and overwrite the index with that of X
            elif output_scitype == "Primitives":
                Xt = pd.concat(Xt)
                Xt.index = X.index
            return Xt

        # input checks and minor coercions on X, y
        ###########################################

        valid, msg, metadata = check_is(
            X, mtype=X_mtype, return_metadata=True, var_name="X"
        )
        if not valid:
            ValueError(msg)

        # checking X
        enforce_univariate = self.get_tag("univariate_only")
        if enforce_univariate and not metadata["is_univariate"]:
            ValueError("X must be univariate but is not")

        # convert X/y to supported inner type, if necessary
        ###################################################

        # subset to the mtypes that are of the same scitype as X/y
        X_inner_mtype = [
            mt for mt in X_inner_mtype if mtype_to_scitype(mt) == X_scitype
        ]

        y_inner_mtype = [
            mt for mt in y_inner_mtype if mtype_to_scitype(mt) == y_scitype
        ]

        # convert X and y to a supported internal type
        #  if X/y type is already supported, no conversion takes place
        X_inner = convert_to(
            X,
            to_type=X_inner_mtype,
            as_scitype=X_scitype,
        )
        y_inner = convert_to(
            y,
            to_type=y_inner_mtype,
            as_scitype=y_scitype,
        )

        # carry out the transformation
        ###################################################

        Xt = self._transform(X=X_inner, y=y_inner)

        # convert transformed X back to input mtype
        ###########################################

        # if we converted Series to "one-instance-Panel", revert that
        if X_was_Series:
            Xt = convert_Panel_to_Series(Xt)

        Xt = convert_to(
            Xt,
            to_type=X_orig_mtype,
            as_scitype=X_orig_scitype,
        )

        return Xt

    def fit_transform(self, X, y=None, Z=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed
        y : Series or Panel, optional (default=None)
            Additional data, e.g., labels for transformation
        Z : possible alias for X; should not be passed when X is passed
            alias Z will be deprecated in version 0.9.0

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported
        """
        X = _handle_alias(X, Z)
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y).transform(X, y)

    # def inverse_transform(self, Z, X=None):
    #     raise NotImplementedError("abstract method")
    #
    # def update(self, Z, X=None, update_params=False):
    #     raise NotImplementedError("abstract method")

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        core logic

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, optional, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        # default fit is "no fitting happens"
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        core logic

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported
        """
        raise NotImplementedError("abstract method")


def _handle_alias(X, Z):
    """Handle Z as an alias for X, return X/Z.

    Parameters
    ----------
    X: any object
    Z: any object

    Returns
    -------
    X if Z is None, Z if X is None

    Raises
    ------
    ValueError both X and Z are not None
    """
    if Z is None:
        return X
    elif X is None:
        warnings.warn(
            "argument Z will be deprecated in transformers, sktime version 0.9.0"
        )
        return Z
    else:
        raise ValueError("X and Z are aliases, at most one of them should be passed")


class _SeriesToPrimitivesTransformer(BaseTransformer):
    """Transformer base class for series to primitive(s) transforms."""

    # class is temporary for downwards compatibility

    # default tag values for "Series-to-Primitives"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
    }


class _SeriesToSeriesTransformer(BaseTransformer):
    """Transformer base class for series to series transforms."""

    # class is temporary for downwards compatibility

    # default tag values for "Series-to-Series"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
    }


class _PanelToTabularTransformer(BaseTransformer):
    """Transformer base class for panel to tabular transforms."""

    # class is temporary for downwards compatibility

    # default tag values for "Series-to-Series"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
    }


class _PanelToPanelTransformer(BaseTransformer):
    """Transformer base class for panel to panel transforms."""

    # class is temporary for downwards compatibility

    # default tag values for "Series-to-Series"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
    }
