# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class template for transformers.

    class name: BaseTransformer

Covers all types of transformers.
Type and behaviour of transformer is determined by the following tags:
    "scitype:transform-input" tag with values "Primitives" or "Series"
        this determines expected type of input of transform
        if "Primitives", expected inputs X are pd.DataFrame
        if "Series", expected inputs X are Series or Panel
        Note: placeholder tag for upwards compatibility
            currently only "Series" is supported
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
from sktime.datatypes import check_is_mtype, convert_to, mtype, mtype_to_scitype
from sktime.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
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
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit-in-transform": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
    }

    # allowed mtypes for transformers - Series and Panel
    ALLOWED_INPUT_MTYPES = [
        "pd.Series",
        "pd.DataFrame",
        "np.ndarray",
        "nested_univ",
        "numpy3D",
        # "numpyflat",
        "pd-multiindex",
        # "pd-wide",
        # "pd-long",
        "df-list",
    ]

    def __init__(self):
        super(BaseTransformer, self).__init__()

    def fit(self, X, y=None, Z=None):
        """Fit transformer to X, optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets is_fitted flag to True.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to fit transform to, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation
        Z : possible alias for X; should not be passed when X is passed
            alias Z will be deprecated in version 0.10.0

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

        # input checks and minor coercions on X, y
        ###########################################

        valid, msg, metadata = check_is_mtype(
            X, mtype=self.ALLOWED_INPUT_MTYPES, return_metadata=True, var_name="X"
        )
        if not valid:
            raise ValueError(msg)

        # checking X
        enforce_univariate = self.get_tag("univariate-only")
        if enforce_univariate and not metadata["is_univariate"]:
            raise ValueError("X must be univariate but is not")

        # retrieve mtypes/scitypes of all objects
        #########################################

        X_input_mtype = mtype(X)
        X_input_scitype = mtype_to_scitype(X_input_mtype)
        y_input_mtype = mtype(y)
        y_input_scitype = mtype_to_scitype(y_input_mtype)

        X_inner_mtype = self.get_tag("X_inner_mtype")
        if not isinstance(X_inner_mtype, list):
            X_inner_mtype = [X_inner_mtype]
        X_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in X_inner_mtype]))

        y_inner_mtype = self.get_tag("y_inner_mtype")
        if not isinstance(y_inner_mtype, list):
            y_inner_mtype = [y_inner_mtype]
        # y_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in y_inner_mtype]))

        # treating Series vs Panel conversion for X
        ###########################################

        # there are three cases to treat:
        # 1. if the internal _fit supports X's scitype, move on to mtype conversion
        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        # 3. internal only has Series but X is Panel: auto-vectorization over instances
        #     currently, this is enabled by conversion to df-list mtype
        #     auto-vectorization is not supported if y is passed
        #       individual estimators that vectorize over y must implement individually

        # 1. nothing to do - simply don't enter any of the ifs below

        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        if X_input_scitype == "Series" and "Series" not in X_inner_scitypes:
            X = convert_Series_to_Panel(X)

        # 3. internal only has Series but X is Panel: loop over instances
        elif X_input_scitype == "Panel" and "Panel" not in X_inner_scitypes:
            if y is not None:
                raise ValueError(
                    "no default behaviour if _fit does not support Panel, "
                    " but X is Panel and y is not None"
                )
            X = convert_to(X, to_type="df-list", as_scitype="Panel")
            # this fits one transformer per instance
            self.transformers_ = [clone(self).fit(Xi) for Xi in X]
            # recurse and leave function - recursion does input checks/conversion
            # also set is_fitted flag to True since we leave function here
            self._is_fitted = True
            return self

        X_mtype = mtype(X)
        X_scitype = mtype_to_scitype(X_mtype)

        # for debugging, exception if the conversion fails (this should never happen)
        if X_scitype not in X_inner_scitypes:
            raise RuntimeError("conversion of X to X_inner unsuccessful, unexpected")

        # convert X/y to supported inner type, if necessary
        ###################################################

        # subset to the mtypes that are of the same scitype as X/y
        X_inner_mtype = [
            mt for mt in X_inner_mtype if mtype_to_scitype(mt) == X_scitype
        ]

        y_inner_mtype = [
            mt for mt in y_inner_mtype if mtype_to_scitype(mt) == y_input_scitype
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
            as_scitype=y_input_scitype,
        )

        # todo: uncomment this once Z is completely gone
        # self._fit(X=X_inner, y=y_inner)
        # less robust workaround until then
        self._fit(X_inner, y_inner)

        self._is_fitted = True
        return self

    def transform(self, X, y=None, Z=None):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation
        Z : possible alias for X; should not be passed when X is passed
            alias Z will be deprecated in version 0.10.0

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |          | `transform`  |                        |
            |   `X`    |  `-output`   |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same mtype
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
        """
        X = _handle_alias(X, Z)

        # check whether is fitted
        self.check_is_fitted()

        # input checks and minor coercions on X, y
        ###########################################

        valid, msg, metadata = check_is_mtype(
            X, mtype=self.ALLOWED_INPUT_MTYPES, return_metadata=True, var_name="X"
        )
        if not valid:
            ValueError(msg)

        # checking X
        enforce_univariate = self.get_tag("univariate-only")
        if enforce_univariate and not metadata["is_univariate"]:
            ValueError("X must be univariate but is not")

        # retrieve mtypes/scitypes of all objects
        #########################################

        X_input_mtype = mtype(X)
        X_input_scitype = mtype_to_scitype(X_input_mtype)
        y_input_mtype = mtype(y)
        y_input_scitype = mtype_to_scitype(y_input_mtype)

        output_scitype = self.get_tag("scitype:transform-output")

        X_inner_mtype = self.get_tag("X_inner_mtype")
        if not isinstance(X_inner_mtype, list):
            X_inner_mtype = [X_inner_mtype]
        X_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in X_inner_mtype]))

        y_inner_mtype = self.get_tag("y_inner_mtype")
        if not isinstance(y_inner_mtype, list):
            y_inner_mtype = [y_inner_mtype]
        # y_inner_scitypes = list(set([mtype_to_scitype(mt) for mt in y_inner_mtype]))

        # treating Series vs Panel conversion for X
        ###########################################

        # there are three cases to treat:
        # 1. if the internal _fit supports X's scitype, move on to mtype conversion
        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        # 3. internal only has Series but X is Panel:  loop over instances
        #     currently this is enabled by conversion to df-list mtype
        #     and this does not support y (unclear what should happen here)

        # 1. nothing to do - simply don't enter any of the ifs below
        #   the "ifs" for case 2 and 3 below are skipped under the condition
        #       X_input_scitype in X_inner_scitypes
        #   case 2 has an "else" which remembers that it wasn't entered

        # 2. internal only has Panel but X is Series: consider X as one-instance Panel
        if (
            X_input_scitype == "Series"
            and "Series" not in X_inner_scitypes
            and "Panel" in X_inner_scitypes
        ):
            # convert the Series X to a one-element Panel
            X = convert_Series_to_Panel(X)
            # remember that we converted the Series to a one-element Panel
            X_was_Series = True
        else:
            # remember that we didn't convert a Series to a one-element Panel
            X_was_Series = False

        # 3. internal only has Series but X is Panel: loop over instances
        if (
            X_input_scitype == "Panel"
            and "Panel" not in X_inner_scitypes
            and "Series" in X_inner_scitypes
        ):
            if y is not None:
                ValueError(
                    "no default behaviour if _fit does not support Panel, "
                    " but X is Panel and y is not None"
                )
            X = convert_to(X, to_type="df-list", as_scitype="Panel")

            # depending on whether fitting happens, apply fitted or unfitted instances
            if not self.get_tag("fit-in-transform"):
                # these are the transformers-per-instanced, fitted in fit
                transformers = self.transformers_
                if len(transformers) != len(X):
                    raise RuntimeError(
                        "found different number of instances in transform than in fit"
                    )

                Xt = [transformers[i].transform(X[i]) for i in range(len(X))]
                # now we have a list of transformed instances
            else:
                # if no fitting happens, just apply transform multiple times
                Xt = [self.transform(X[i]) for i in range(len(X))]

            # if the output is Series, Xt is a Panel and we convert back
            if output_scitype == "Series":
                Xt = convert_to(Xt, to_type=X_input_mtype, as_scitype="Panel")

            # if the output is Primitives, we have a list of one-row dataframes
            # we concatenate those and overwrite the index with that of X
            elif output_scitype == "Primitives":
                Xt = pd.concat(Xt)
                Xt = Xt.reset_index(drop=True)
            return Xt

        # convert X/y to supported inner type, if necessary
        ###################################################

        # variables for the scitype of the current X (possibly converted)
        #     y wasn't converted so we can use y_input_scitype
        X_mtype = mtype(X)
        X_scitype = mtype_to_scitype(X_mtype)

        # subset to the mtypes that are of the same scitype as X/y
        X_inner_mtype = [
            mt for mt in X_inner_mtype if mtype_to_scitype(mt) == X_scitype
        ]

        y_inner_mtype = [
            mt for mt in y_inner_mtype if mtype_to_scitype(mt) == y_input_scitype
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
            as_scitype=y_input_scitype,
        )

        # carry out the transformation
        ###################################################

        # todo: uncomment this once Z is completely gone
        # Xt = self._transform(X=X_inner, y=y_inner)
        # less robust workaround until then
        Xt = self._transform(X_inner, y_inner)

        # convert transformed X back to input mtype
        ###########################################

        # if we converted Series to "one-instance-Panel", revert that
        if X_was_Series and output_scitype == "Series":
            Xt = convert_Panel_to_Series(Xt)

        if output_scitype == "Series":
            # if the transformer outputs multivariate series,
            #   we cannot convert back to pd.Series, do pd.DataFrame instead then
            _, _, metadata = check_is_mtype(
                Xt, ["pd.DataFrame", "pd.Series", "np.ndarray"], return_metadata=True
            )
            if not metadata["is_univariate"] and X_input_mtype == "pd.Series":
                X_output_mtype = "pd.DataFrame"
            else:
                X_output_mtype = X_input_mtype
            Xt = convert_to(
                Xt,
                to_type=X_output_mtype,
                as_scitype=X_input_scitype,
            )
        elif output_scitype == "Primitives":
            # we "abuse" the Series converter to ensure df output
            # & reset index to have integers for instances
            if isinstance(Xt, (pd.DataFrame, pd.Series)):
                Xt = Xt.reset_index(drop=True)
            Xt = convert_to(
                Xt,
                to_type="pd.DataFrame",
                as_scitype="Series",
            )
        else:
            # output_scitype is "Panel" and no need for conversion
            pass

        return Xt

    def fit_transform(self, X, y=None, Z=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets is_fitted flag to True.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation
        Z : possible alias for X; should not be passed when X is passed
            alias Z will be deprecated in version 0.10.0

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

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same mtype
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
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
        y : Series or Panel of mtype y_inner_mtype, default=None
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
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |          | `transform`  |                        |
            |   `X`    |  `-output`   |     type of return     |
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
            "argument Z will be deprecated in transformers, sktime version 0.10.0",
            category=DeprecationWarning,
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
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
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
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
    }


class _PanelToTabularTransformer(BaseTransformer):
    """Transformer base class for panel to tabular transforms."""

    # class is temporary for downwards compatibility

    # default tag values for "Series-to-Series"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
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
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
    }
