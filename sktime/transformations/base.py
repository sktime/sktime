# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for transformers.

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

__author__ = ["mloning", "fkiraly", "miraep8"]
__all__ = [
    "BaseTransformer",
    "_SeriesToPrimitivesTransformer",
    "_SeriesToSeriesTransformer",
    "_PanelToTabularTransformer",
    "_PanelToPanelTransformer",
]

from itertools import product
from typing import Union

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.datatypes import (
    VectorizedDF,
    check_is_error_msg,
    check_is_mtype,
    check_is_scitype,
    convert,
    convert_to,
    mtype_to_scitype,
    update_data,
)
from sktime.datatypes._dtypekind import DtypeKind
from sktime.datatypes._series_as_panel import convert_to_scitype
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.sklearn import (
    is_sklearn_classifier,
    is_sklearn_clusterer,
    is_sklearn_regressor,
    is_sklearn_transformer,
)

# single/multiple primitives
Primitive = Union[np.integer, int, float, str]
Primitives = np.ndarray

# tabular/cross-sectional data
Tabular = Union[pd.DataFrame, np.ndarray]  # 2d arrays

# univariate/multivariate series
UnivariateSeries = Union[pd.Series, np.ndarray]
MultivariateSeries = Union[pd.DataFrame, np.ndarray]
Series = Union[UnivariateSeries, MultivariateSeries]

# panel/longitudinal/series-as-features data
Panel = Union[pd.DataFrame, np.ndarray]  # 3d or nested array


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseTransformer(BaseEstimator):
    """Transformer base class."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "transformer",  # type of object
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "capability:inverse_transform:range": None,
        "capability:inverse_transform:exact": True,
        # inverting range of inverse transform = domain of invertibility of transform
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_X": True,  # does X need to be passed in fit?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        "capability:missing_values": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "capability:categorical_in_X": False,
        # does the transformer apply hierarchical reconciliation?
        "remember_data": False,  # whether all data seen is remembered as self._X
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
    }

    # default config values
    # see set_config documentation for details
    _config = {
        "input_conversion": "on",
        # controls input checks and conversions,
        #  for _fit, _transform, _inverse_transform, _update
        # valid values:
        # "on" - input check and conversion is carried out
        # "off" - input check and conversion is not done before passing to inner methods
        # valid mtype string - input is assumed to specified mtype
        "output_conversion": "on",
        # controls output conversion for _transform, _inverse_transform
        # valid values:
        # "on" - if input_conversion is "on", output conversion is carried out
        # "off" - output of _transform, _inverse_transform is directly returned
        # valid mtype string - output is converted to specified mtype
        "backend:parallel": None,  # parallelization backend for broadcasting
        #  {None, "dask", "loky", "multiprocessing", "threading","ray"}
        #  None: no parallelization
        #  "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        #  "joblib": uses custom joblib backend, set via `joblib_backend` tag
        #  "dask": uses `dask`, requires `dask` package in environment
        #  "ray": uses ``ray``, requires `ray` package in environment
        "backend:parallel:params": None,  # params for parallelization backend
    }

    _config_doc = {
        "input_conversion": """
        input_conversion : str, one of "on" (default), "off", or valid mtype string
            controls input checks and conversions,
            for ``_fit``, ``_transform``, ``_inverse_transform``, ``_update``

            * ``"on"`` - input check and conversion is carried out
            * ``"off"`` - input check and conversion are not carried out
              before passing data to inner methods
            * valid mtype string - input is assumed to specified mtype,
              conversion is carried out but no check
        """,
        "output_conversion": """
        output_conversion : str, one of "on", "off", valid mtype string
            controls output conversion for ``_transform``, ``_inverse_transform``

            * ``"on"`` - if input_conversion is "on", output conversion is carried out
            * ``"off"`` - output of ``_transform``, ``_inverse_transform``
              is directly returned
            * valid mtype string - output is converted to specified mtype
        """,
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
        "pd_multiindex_hier",
    ]

    def __init__(self):
        self._converter_store_X = dict()  # storage dictionary for in/output conversion

        super().__init__()
        _check_estimator_deps(self)

    def _is_transformer(self, other):
        """Check whether other is a transformer - sklearn or sktime.

        Returns True iff at least one of the following is True:

        * ``is_sklearn_transformer(other)``
        * ``scitype(other) == "transformer"``

        Parameters
        ----------
        other : object
            object to check
        """
        from sktime.registry import is_scitype

        is_sktime_transformr = is_scitype(other, "transformer")
        return is_sklearn_transformer(other) or is_sktime_transformr

    def __mul__(self, other):
        """Magic * method, return (right) concatenated TransformerPipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: ``sktime`` or ``sklearn`` compatible transformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-TransformerPipeline `sktime` transformers
        """
        from sktime.transformations.compose import TransformerPipeline

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformerPipeline does the rest, e.g., case distinctions on other
        if (
            self._is_transformer(other)
            or is_sklearn_classifier(other)
            or is_sklearn_clusterer(other)
            or is_sklearn_regressor(other)
        ):
            self_as_pipeline = TransformerPipeline(steps=[self])
            return self_as_pipeline * other
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated TransformerPipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: ``sktime`` or ``sklearn`` compatible transformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `other` (first) with `self` (last).
            not nested, contains only non-TransformerPipeline `sktime` transformers
        """
        from sktime.transformations.compose import TransformerPipeline

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformerPipeline does the rest, e.g., case distinctions on other
        if self._is_transformer(other):
            self_as_pipeline = TransformerPipeline(steps=[self])
            return other * self_as_pipeline
        else:
            return NotImplemented

    def __or__(self, other):
        """Magic | method, return MultiplexTransformer.

        Implemented for `other` being either a MultiplexTransformer or a transformer.

        Parameters
        ----------
        other: ``sktime`` or ``sklearn`` compatible transformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        MultiplexTransformer object
        """
        from sktime.transformations.compose import MultiplexTransformer

        if self._is_transformer(other):
            multiplex_self = MultiplexTransformer([self])
            return multiplex_self | other
        else:
            return NotImplemented

    def __add__(self, other):
        """Magic + method, return (right) concatenated FeatureUnion.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: ``sktime`` or ``sklearn`` compatible transformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        FeatureUnion object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-TransformerPipeline `sktime` transformers
        """
        from sktime.transformations.compose import FeatureUnion

        # we wrap self in a pipeline, and concatenate with the other
        #   the FeatureUnion does the rest, e.g., case distinctions on other
        if self._is_transformer(other):
            self_as_pipeline = FeatureUnion(transformer_list=[self])
            return self_as_pipeline + other
        else:
            return NotImplemented

    def __radd__(self, other):
        """Magic + method, return (left) concatenated FeatureUnion.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: ``sktime`` or ``sklearn`` compatible transformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        FeatureUnion object, concatenation of `other` (first) with `self` (last).
            not nested, contains only non-FeatureUnion `sktime` transformers
        """
        from sktime.transformations.compose import FeatureUnion

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformerPipeline does the rest, e.g., case distinctions on other
        if self._is_transformer(other):
            self_as_pipeline = FeatureUnion(transformer_list=[self])
            return other + self_as_pipeline
        else:
            return NotImplemented

    def __invert__(self):
        """Magic unary ~ (inversion) method, return InvertTransform of self.

        Returns
        -------
        `InvertTransform` object, containing `self`.
        """
        from sktime.transformations.compose import InvertTransform

        return InvertTransform(self)

    def __neg__(self):
        """Magic unary - (negation) method, return OptionalPassthrough of self.

        Intuition: `OptionalPassthrough` is "not having transformer", as an option.

        Returns
        -------
        `OptionalPassthrough` object, containing `self`, with `passthrough=False`.
            The `passthrough` parameter can be set via `set_params`.
        """
        from sktime.transformations.compose import OptionalPassthrough

        return OptionalPassthrough(self, passthrough=False)

    def __getitem__(self, key):
        """Magic [...] method, return column subsetted transformer.

        First index does input subsetting, second index does output subsetting.

        Keys must be valid inputs for `columns` in `ColumnSelect`.

        Parameters
        ----------
        key: valid input for `columns` in `ColumnSelect`, or pair thereof
            keys can also be a :-slice, in which case it is considered as not passed

        Returns
        -------
        the following TransformerPipeline object:
            ColumnSelect(columns1) * self * ColumnSelect(columns2)
            where `columns1` is first or only item in `key`, and `columns2` is the last
            if only one item is passed in `key`, only `columns1` is applied to input
        """
        from sktime.transformations.series.subset import ColumnSelect

        def is_noneslice(obj):
            res = isinstance(obj, slice)
            res = res and obj.start is None and obj.stop is None and obj.step is None
            return res

        if isinstance(key, tuple):
            if not len(key) == 2:
                raise ValueError(
                    "there should be one or two keys when calling [] or getitem, "
                    "e.g., mytrafo[key], or mytrafo[key1, key2]"
                )
            columns1 = key[0]
            columns2 = key[1]
            if is_noneslice(columns1) and is_noneslice(columns2):
                return self
            elif is_noneslice(columns2):
                return ColumnSelect(columns1) * self
            elif is_noneslice(columns1):
                return self * ColumnSelect(columns2)
            else:
                return ColumnSelect(columns1) * self * ColumnSelect(columns2)
        else:
            return ColumnSelect(key) * self

    def fit(self, X, y=None):
        """Fit transformer to X, optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:

            * Sets fitted model attributes ending in "_", fitted attributes are
              inspectable via ``get_fitted_params``.
            * Sets ``self.is_fitted`` flag to ``True``.
            * if ``self.get_tag("remember_data")`` is ``True``, memorizes X as
              ``self._X``, coerced to ``self.get_tag("X_inner_mtype")``.

        Parameters
        ----------
        X : time series in ``sktime`` compatible data container format
            Data to fit transform to.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection of time series.
              ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see transformer tutorial ``examples/03_transformers.ipynb``

        y : optional, data in sktime compatible data format, default=None
            Additional data, e.g., labels for transformation
            If ``self.get_tag("requires_y")`` is ``True``,
            must be passed in ``fit``, not optional.
            For required format, see class docstring for details.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # if fit is called, estimator is reset, including fitted state
        self.reset()

        # skip everything if fit_is_empty is True and we do not need to remember data
        if self.get_tag("fit_is_empty") and not self.get_tag("remember_data", False):
            self._is_fitted = True
            self._is_vectorized = "unknown"
            return self

        # if requires_X is set, X is required in fit and update
        if self.get_tag("requires_X") and X is None:
            raise ValueError(f"{self.__class__.__name__} requires `X` in `fit`.")

        # if requires_y is set, y is required in fit and update
        if self.get_tag("requires_y") and y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` in `fit`.")

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # memorize X as self._X, if remember_data tag is set to True
        if self.get_tag("remember_data", False):
            self._X = update_data(None, X_new=X_inner)

        # skip the rest if fit_is_empty is True
        if self.get_tag("fit_is_empty"):
            self._is_fitted = True
            return self

        # checks and conversions complete, pass to inner fit
        #####################################################
        vectorization_needed = isinstance(X_inner, VectorizedDF)
        self._is_vectorized = vectorization_needed
        # we call the ordinary _fit if no looping/vectorization needed
        if not vectorization_needed:
            self._fit(X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of fit
            self._vectorize("fit", X=X_inner, y=y_inner)

        # this should happen last: fitted state is set to True
        self._is_fitted = True

        return self

    def transform(self, X, y=None):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.is_fitted``, must be True

        Parameters
        ----------
        X : time series in ``sktime`` compatible data container format
            Data to transform.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection of time series.
              ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see transformer tutorial ``examples/03_transformers.ipynb``

        y : optional, data in sktime compatible data format, default=None
            Additional data, e.g., labels for transformation.
            Some transformers require this, see class docstring for details.

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:

        .. list-table::
            :widths: 35 35 40
            :header-rows: 2

            * -
              - `transform`
              -
            * - `X`
              - `-output`
              - type of return
            * - `Series`
              - `Primitives`
              - `pd.DataFrame` (1-row)
            * - `Panel`
              - `Primitives`
              - `pd.DataFrame`
            * - `Series`
              - `Series`
              - `Series`
            * - `Panel`
              - `Series`
              - `Panel`
            * - `Series`
              - `Panel`
              - `Panel`

        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:

            * if ``X`` is ``Series`` (e.g., ``pd.DataFrame``)
            and ``transform-output`` is ``Series``,
            then the return is a single `Series` of the same mtype.
            Example: detrending a single series

            * if ``X`` is ``Panel`` (e.g., ``pd-multiindex``) and ``transform-output``
            is ``Series``,
            then the return is `Panel` with same number of instances as ``X``
            (the transformer is applied to each input Series instance).
            Example: all series in the panel are detrended individually

            * if ``X`` is ``Series`` or ``Panel`` and ``transform-output`` is
            ``Primitives``,
            then the return is ``pd.DataFrame`` with as many rows as instances in ``X``
            Example: i-th row of the return has mean and variance of the i-th series

            * if ``X`` is ``Series`` and ``transform-output`` is ``Panel``,
            then the return is a ``Panel`` object of type ``pd-multiindex``.
            Example: i-th instance of the output is the i-th window running over ``X``
        """
        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        # check if we need to vectorize
        if getattr(self, "_is_vectorized", "unknown") == "unknown":
            vectorization_needed = isinstance(X_inner, VectorizedDF)
        else:
            vectorization_needed = self._is_vectorized

        # if no vectorization needed, we call _transform directly
        if not vectorization_needed:
            Xt = self._transform(X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of predict
            Xt = self._vectorize("transform", X=X_inner, y=y_inner)

        # obtain configs to control input and output control
        configs = self.get_config()
        input_conv = configs["input_conversion"]
        output_conv = configs["output_conversion"]

        # convert to output mtype
        if X is None or Xt is None:
            X_out = Xt
        elif input_conv and output_conv:
            X_out = self._convert_output(Xt, metadata=metadata)
        else:
            X_out = Xt

        return X_out

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
            possibly coerced to inner type or update_data compatible type
            by reference, when possible
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : time series in ``sktime`` compatible data container format
            Data to fit transform to, and data to transform.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection of time series.
              ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see transformer tutorial ``examples/03_transformers.ipynb``

        y : optional, data in sktime compatible data format, default=None
            Additional data, e.g., labels for transformation
            If ``self.get_tag("requires_y")`` is ``True``,
            must be passed in ``fit``, not optional.
            For required format, see class docstring for details.

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

            * if ``X`` is ``Series`` (e.g., ``pd.DataFrame``)
            and ``transform-output`` is ``Series``,
            then the return is a single `Series` of the same mtype.
            Example: detrending a single series

            * if ``X`` is ``Panel`` (e.g., ``pd-multiindex``) and ``transform-output``
            is ``Series``,
            then the return is `Panel` with same number of instances as ``X``
            (the transformer is applied to each input Series instance).
            Example: all series in the panel are detrended individually

            * if ``X`` is ``Series`` or ``Panel`` and ``transform-output`` is
            ``Primitives``,
            then the return is ``pd.DataFrame`` with as many rows as instances in ``X``
            Example: i-th row of the return has mean and variance of the i-th series

            * if ``X`` is ``Series`` and ``transform-output`` is ``Panel``,
            then the return is a ``Panel`` object of type ``pd-multiindex``.
            Example: i-th instance of the output is the i-th window running over ``X``
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        Currently it is assumed that only transformers with tags
            "scitype:transform-input"="Series", "scitype:transform-output"="Series",
        have an inverse_transform.

        State required:
            Requires state to be "fitted".

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.is_fitted``, must be True

        Parameters
        ----------
        X : time series in ``sktime`` compatible data container format
            Data to fit transform to.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection of time series.
              ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see transformer tutorial ``examples/03_transformers.ipynb``

        y : optional, data in sktime compatible data format, default=None
            Additional data, e.g., labels for transformation.
            Some transformers require this, see class docstring for details.

        Returns
        -------
        inverse transformed version of X
            of the same type as X, and conforming to mtype format specifications
        """
        if self.get_tag("skip-inverse-transform"):
            return X

        if not self.get_tag("capability:inverse_transform"):
            raise NotImplementedError(
                f"{type(self)} does not implement inverse_transform"
            )

        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        # check if we need to vectorize
        if getattr(self, "_is_vectorized", "unknown") == "unknown":
            vectorization_needed = isinstance(X_inner, VectorizedDF)
        else:
            vectorization_needed = self._is_vectorized

        # if no vectorization needed, we call _inverse_transform directly
        if not vectorization_needed:
            # capture edge condition where:
            # transformer is univariate, transform produces multivariate
            # in this case the check_X_y will convert to VectorizedDF,
            # but inverse_transform expects a DataFrame
            # example: time series decomposition algorithms
            if isinstance(X_inner, VectorizedDF):
                X_inner = X_inner.X_multiindex
            Xt = self._inverse_transform(X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of predict
            Xt = self._vectorize("inverse_transform", X=X_inner, y=y_inner)

        # convert to output mtype
        configs = self.get_config()
        output_conv = configs["output_conversion"]

        if output_conv != "off":
            X_out = self._convert_output(Xt, metadata=metadata, inverse=True)
        else:
            X_out = Xt

        return X_out

    def update(self, X, y=None, update_params=True):
        """Update transformer with X, optionally y.

        State required:
            Requires state to be "fitted".

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.is_fitted``, must be True

        Writes to self:

            * Fitted model attributes ending in "_".
            * if ``remember_data`` tag is True, writes to ``self._X``,
              updated by values in ``X``, via ``update_data``.

        Parameters
        ----------
        X : time series in ``sktime`` compatible data container format
            Data to update transformation with

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection of time series.
              ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see transformer tutorial ``examples/03_transformers.ipynb``

        y : optional, data in sktime compatible data format, default=None
            Additional data, e.g., labels for transformation.
            Some transformers require this, see class docstring for details.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # check whether is fitted
        self.check_is_fitted()

        # if requires_y is set, y is required in fit and update
        if self.get_tag("requires_y") and y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` in `update`.")

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update memory of X, if remember_data tag is set to True
        if self.get_tag("remember_data", False):
            self._X = update_data(None, X_new=X_inner)

        # skip everything if update_params is False
        # skip everything if fit_is_empty is True
        if not update_params or self.get_tag("fit_is_empty", False):
            return self

        # checks and conversions complete, pass to inner fit
        #####################################################
        vectorization_needed = isinstance(X_inner, VectorizedDF)
        # we call the ordinary _fit if no looping/vectorization needed
        if not vectorization_needed:
            self._update(X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of fit
            self._vectorize("update", X=X_inner, y=y_inner)

        return self

    def get_fitted_params(self, deep=True):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        deep : bool, default=True
            Whether to return fitted parameters of components.

            * If True, will return a dict of parameter name : value for this object,
              including fitted parameters of fittable components
              (= BaseEstimator-valued parameters).
            * If False, will return a dict of parameter name : value for this object,
              but not include fitted parameters of components.

        Returns
        -------
        fitted_params : dict with str-valued keys
            Dictionary of fitted parameters, paramname : paramvalue
            keys-value pairs include:

            * always: all fitted parameters of this object, as via `get_param_names`
              values are fitted parameter value for that key, of this object
            * if `deep=True`, also contains keys/value pairs of component parameters
              parameters of components are indexed as `[componentname]__[paramname]`
              all parameters of `componentname` appear as `paramname` with its value
            * if `deep=True`, also contains arbitrary levels of component recursion,
              e.g., `[componentname]__[componentcomponentname]__[paramname]`, etc
        """
        # if self is not vectorized, run the default get_fitted_params
        # the condition is: _is_vectorized is boolean, False, or "unknown"
        is_vectorized = getattr(self, "_is_vectorized", False)
        is_not_vectorized = isinstance(is_vectorized, bool) and not is_vectorized
        is_not_vectorized = is_not_vectorized or is_vectorized == "unknown"

        if is_not_vectorized:
            return super().get_fitted_params(deep=deep)

        # otherwise, we delegate to the instances' get_fitted_params
        # instances' parameters are returned at dataframe-slice-like keys
        fitted_params = {}

        # transformers contains a pd.DataFrame with the individual transformers
        transformers = self.transformers_

        # return transformers in the "transformers" param
        fitted_params["transformers"] = transformers

        def _to_str(x):
            if isinstance(x, str):
                x = f"'{x}'"
            return str(x)

        # populate fitted_params with transformers and their parameters
        for ix, col in product(transformers.index, transformers.columns):
            trafo = transformers.loc[ix, col]
            trafo_key = f"transformers.loc[{_to_str(ix)},{_to_str(col)}]"
            fitted_params[trafo_key] = trafo
            trafo_params = trafo.get_fitted_params(deep=deep)
            for key, val in trafo_params.items():
                fitted_params[f"{trafo_key}__{key}"] = val

        return fitted_params

    def _check_X_y(self, X=None, y=None, return_metadata=False):
        """Check and coerce X/y for fit/transform functions.

        Parameters
        ----------
        X : object of sktime compatible time series type
            can be Series, Panel, Hierarchical
        y : None (default), or object of sktime compatible time series type
            can be Series, Panel, Hierarchical
        return_metadata : bool, optional, default=False
            whether to return the metadata return object

        Returns
        -------
        X_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("X_inner_mtype") format
            Case 1: self.get_tag("X_inner_mtype") supports scitype of X, then
                converted/coerced version of X, mtype determined by "X_inner_mtype" tag
            Case 2: self.get_tag("X_inner_mtype") supports *higher* scitype than X
                then X converted to "one-Series" or "one-Panel" sub-case of that scitype
                always pd-multiindex (Panel) or pd_multiindex_hier (Hierarchical)
            Case 3: self.get_tag("X_inner_mtype") supports only *simpler* scitype than X
                then VectorizedDF of X, iterated as the most complex supported scitype
        y_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("y_inner_mtype") format
            Case 1: self.get_tag("y_inner_mtype") supports scitype of y, then
                converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            Case 2: self.get_tag("y_inner_mtype") supports *higher* scitype than y
                then X converted to "one-Series" or "one-Panel" sub-case of that scitype
                always pd-multiindex (Panel) or pd_multiindex_hier (Hierarchical)
            Case 3: self.get_tag("y_inner_mtype") supports only *simpler* scitype than y
                then VectorizedDF of X, iterated as the most complex supported scitype
            Case 4: None if y was None, or self.get_tag("y_inner_mtype") is "None"

            Complexity order above: Hierarchical > Panel > Series

        metadata : dict, returned only if return_metadata=True
            dictionary with str keys, contents as follows
            _converter_store_X : dict, metadata from X conversion, for back-conversion
            _X_mtype_last_seen : str, mtype of X seen last
            _X_input_scitype : str, scitype of X seen last
            _convert_case : str, conversion case (see above), one of
                "case 1: scitype supported"
                "case 2: higher scitype supported"
                "case 3: requires vectorization"

        Raises
        ------
        TypeError if X is None
        TypeError if X or y is not one of the permissible Series mtypes
        TypeError if X is not compatible with self.get_tag("univariate_only")
            if tag value is "True", X must be univariate
        ValueError if self.get_tag("requires_y")=True but y is None
        """
        if X is None:
            if return_metadata:
                return X, y, {}
            else:
                return X, y

        # skip conversion if it is turned off
        if self.get_config()["input_conversion"] != "on":
            if return_metadata:
                return X, y, None
            else:
                return X, y

        metadata = dict()
        metadata["_converter_store_X"] = dict()

        def _most_complex_scitype(scitypes, smaller_equal_than=None):
            """Return most complex scitype in a list of str."""
            if "Hierarchical" in scitypes and smaller_equal_than == "Hierarchical":
                return "Hierarchical"
            elif "Panel" in scitypes and smaller_equal_than != "Series":
                return "Panel"
            elif "Series" in scitypes:
                return "Series"
            elif smaller_equal_than is not None:
                return _most_complex_scitype(scitypes)
            else:
                raise ValueError(
                    f"Error in {type(self).__name__}, no series scitypes supported, "
                    "likely a bug in estimator: scitypes arg passed to "
                    f"_most_complex_scitype are {scitypes}"
                )

        def _scitype_A_higher_B(scitypeA, scitypeB):
            """Compare two scitypes regarding complexity."""
            if scitypeA == "Series":
                return False
            if scitypeA == "Panel" and scitypeB == "Series":
                return True
            if scitypeA == "Hierarchical" and scitypeB != "Hierarchical":
                return True
            return False

        # retrieve supported mtypes
        X_inner_mtype = _coerce_to_list(self.get_tag("X_inner_mtype"))
        y_inner_mtype = _coerce_to_list(self.get_tag("y_inner_mtype"))
        X_inner_scitype = mtype_to_scitype(X_inner_mtype, return_unique=True)
        y_inner_scitype = mtype_to_scitype(y_inner_mtype, return_unique=True)

        ALLOWED_MTYPES = self.ALLOWED_INPUT_MTYPES

        # checking X
        X_metadata_required = ["is_univariate", "feature_kind"]

        X_valid, msg, X_metadata = check_is_mtype(
            X,
            mtype=ALLOWED_MTYPES,
            return_metadata=X_metadata_required,
            var_name="X",
        )

        # raise informative error message if X is in wrong format
        allowed_msg = (
            f"Allowed scitypes for X in transformations are "
            f"Series, Panel or Hierarchical, "
            f"for instance a pandas.DataFrame with sktime compatible time indices, "
            f"or with MultiIndex and last(-1) level an sktime compatible time index. "
            f"Allowed compatible mtype format specifications are: {ALLOWED_MTYPES} ."
        )
        msg_start = f"Unsupported input data type in {self.__class__.__name__}, input "
        msg_X = msg_start + "X"
        if not X_valid:
            msg = {k: v for k, v in msg.items() if k in ALLOWED_MTYPES}
            check_is_error_msg(
                msg, var_name=msg_X, allowed_msg=allowed_msg, raise_exception=True
            )

        if DtypeKind.CATEGORICAL in X_metadata["feature_kind"] and not self.get_tag(
            "capability:categorical_in_X"
        ):
            raise TypeError(
                f"Transformer {self} does not support categorical features in X."
            )

        X_scitype = X_metadata["scitype"]
        X_mtype = X_metadata["mtype"]

        # remember these for potential back-conversion (in transform etc)
        metadata["_X_mtype_last_seen"] = X_mtype
        metadata["_X_input_scitype"] = X_scitype

        if X_scitype in X_inner_scitype:
            case = "case 1: scitype supported"
            req_vec_because_rows = False
        elif any(_scitype_A_higher_B(x, X_scitype) for x in X_inner_scitype):
            case = "case 2: higher scitype supported"
            req_vec_because_rows = False
        else:
            case = "case 3: requires vectorization"
            req_vec_because_rows = True
        metadata["_convert_case"] = case

        # checking X vs tags
        inner_univariate = self.get_tag("univariate-only")
        # we remember whether we need to vectorize over columns, and at all
        req_vec_because_cols = inner_univariate and not X_metadata["is_univariate"]
        requires_vectorization = req_vec_because_rows or req_vec_because_cols
        # end checking X

        if y_inner_mtype != ["None"] and y is not None:
            if "Table" in y_inner_scitype:
                y_possible_scitypes = "Table"
            elif X_scitype == "Series":
                y_possible_scitypes = "Series"
            elif X_scitype == "Panel":
                y_possible_scitypes = "Panel"
            elif X_scitype == "Hierarchical":
                y_possible_scitypes = ["Panel", "Hierarchical"]

            y_valid, msg, y_metadata = check_is_scitype(
                y,
                scitype=y_possible_scitypes,
                return_metadata=["feature_kind"],
                var_name="y",
            )

            y_required = self.get_tag("requires_y")

            # raise informative error message if y is is in wrong format
            if not y_valid and y_required:
                allowed_msg = (
                    f"Allowed scitypes for y in transformations depend on X passed. "
                    f"Passed X scitype was {X_scitype}, "
                    f"so allowed scitypes for y are {y_possible_scitypes}. "
                )
                msg_y = msg_start + "y"
                check_is_error_msg(
                    msg, var_name=msg_y, allowed_msg=allowed_msg, raise_exception=True
                )

            elif not y_valid and not y_required:
                # if y is wrong type, we do not pass it to inner methods
                y_scitype = None
                y_inner_mtype = ["None"]
            else:  # y_valid, (y_required does not matter then, we pass y)
                y_scitype = y_metadata["scitype"]
                y_mtype = y_metadata["mtype"]

                if DtypeKind.CATEGORICAL in y_metadata["feature_kind"]:
                    raise TypeError(
                        "Transformers do not support categorical features in y."
                    )

        else:
            # y_scitype is used below - set to None if y is None
            y_scitype = None
        # end checking y

        # no compatibility checks between X and y
        # end compatibility checking X and y

        # convert X & y to supported inner type, if necessary
        #####################################################

        # convert X and y to a supported internal mtype
        #  it X/y mtype is already supported, no conversion takes place
        #  if X/y is None, then no conversion takes place (returns None)
        #  if vectorization is required, we wrap in VectorizedDF

        # case 2. internal only has higher scitype, e.g., inner is Panel and X Series
        #       or inner is Hierarchical and X is Panel or Series
        #   then, consider X as one-instance Panel or Hierarchical
        if case == "case 2: higher scitype supported":
            if X_scitype == "Series" and "Panel" in X_inner_scitype:
                as_scitype = "Panel"
            else:
                as_scitype = "Hierarchical"
            X, X_mtype = convert_to_scitype(
                X, to_scitype=as_scitype, from_scitype=X_scitype, return_to_mtype=True
            )
            X_scitype = as_scitype
            # then pass to case 1, which we've reduced to, X now has inner scitype

        # case 1. scitype of X is supported internally
        # case in ["case 1: scitype supported", "case 2: higher scitype supported"]
        #   and does not require vectorization because of cols (multivariate)
        if not requires_vectorization:
            # converts X
            X_inner = convert(
                X,
                from_type=X_mtype,
                to_type=X_inner_mtype,
                store=metadata["_converter_store_X"],
                store_behaviour="reset",
            )

            # converts y, returns None if y is None
            if y_inner_mtype != ["None"] and y is not None:
                y_inner = convert(
                    y,
                    from_type=y_mtype,
                    to_type=y_inner_mtype,
                    as_scitype=y_scitype,
                )
            else:
                y_inner = None

        # case 3. scitype of X is not supported, only lower complexity one is
        #   then apply vectorization, loop method execution over series/panels
        # elif case == "case 3: requires vectorization":
        else:  # if requires_vectorization
            iterate_X = _most_complex_scitype(X_inner_scitype, X_scitype)
            X_inner = VectorizedDF(
                X=X,
                iterate_as=iterate_X,
                is_scitype=X_scitype,
                iterate_cols=req_vec_because_cols,
            )
            # we also assume that y must be vectorized in this case
            if y_inner_mtype != ["None"] and y is not None:
                # raise ValueError(
                #     f"{type(self).__name__} does not support Panel X if y is not "
                #     f"None, since {type(self).__name__} supports only Series. "
                #     "Auto-vectorization to extend Series X to Panel X can only be "
                #     'carried out if y is None, or "y_inner_mtype" tag is "None". '
                #     "Consider extending _fit and _transform to handle the following "
                #     "input types natively: Panel X and non-None y."
                # )
                iterate_y = _most_complex_scitype(y_inner_scitype, y_scitype)
                y_inner = VectorizedDF(X=y, iterate_as=iterate_y, is_scitype=y_scitype)
            else:
                y_inner = None

        if return_metadata:
            return X_inner, y_inner, metadata
        else:
            return X_inner, y_inner

    def _check_X(self, X=None):
        """Shorthand for _check_X_y with one argument X, see _check_X_y."""
        return self._check_X_y(X=X)[0]

    def _convert_output(self, X, metadata, inverse=False):
        """Convert transform or inverse_transform output to expected format.

        Parameters
        ----------
        X : output of _transform or _vectorize("transform"), or inverse variants
        metadata : dict, output of _check_X_y
        inverse : bool, optional, default = False
            whether conversion is for transform (False) or inverse_transform (True)

        Returns
        -------
        Xt : final output of transform or inverse_transform
        """
        # skip conversion if not both conversions are switched on
        configs = self.get_config()
        input_conv = configs["input_conversion"]
        output_conv = configs["output_conversion"]
        if input_conv != "on" or output_conv != "on":
            return X

        Xt = X
        X_input_mtype = metadata["_X_mtype_last_seen"]
        X_input_scitype = metadata["_X_input_scitype"]
        case = metadata["_convert_case"]
        _converter_store_X = metadata["_converter_store_X"]

        if inverse:
            # the output of inverse transform is equal to input of transform
            output_scitype = self.get_tag("scitype:transform-input")
        else:
            output_scitype = self.get_tag("scitype:transform-output")

        # if we converted Series to "one-instance-Panel/Hierarchical",
        #   or Panel to "one-instance-Hierarchical", then revert that
        # remainder is as in case 1
        #   skipped for output_scitype = "Primitives"
        #       since then the output always is a pd.DataFrame
        if case == "case 2: higher scitype supported" and output_scitype == "Series":
            if self.get_tag("scitype:transform-input") == "Panel":
                # Conversion from Series to Panel done for being compatible with
                # algorithm. Thus, the returned Series should stay a Series.
                pass
            else:
                Xt = convert_to(
                    Xt,
                    to_type=[
                        "pd-multiindex",
                        "numpy3D",
                        "df-list",
                        "pd_multiindex_hier",
                    ],
                )
            Xt = convert_to_scitype(Xt, to_scitype=X_input_scitype)

        # now, in all cases, Xt is in the right scitype,
        #   but not necessarily in the right mtype.
        # additionally, Primitives may have an extra column

        #   "case 1: scitype supported"
        #   "case 2: higher scitype supported"
        #   "case 3: requires vectorization"

        if output_scitype == "Series":
            # output mtype is input mtype
            X_output_mtype = X_input_mtype
            # exception to this: if the transformer outputs multivariate series,
            #   we cannot convert back to pd.Series, do pd.DataFrame instead then
            #   this happens only for Series, not Panel
            if X_input_scitype == "Series":
                if X_input_mtype == "pd.Series":
                    Xt_metadata_required = ["is_univariate"]
                else:
                    Xt_metadata_required = []

                ALLOWED_OUT_MTYPES = ["pd.DataFrame", "pd.Series", "np.ndarray"]
                Xt_valid, Xt_msg, metadata = check_is_mtype(
                    Xt,
                    ALLOWED_OUT_MTYPES,
                    msg_return_dict="dict",
                    return_metadata=Xt_metadata_required,
                )

                if not Xt_valid:
                    Xtd = {k: v for k, v in Xt_msg.items() if k in ALLOWED_OUT_MTYPES}
                    msg_start = (
                        f"Type checking error in output of _transform of "
                        f"{self.__class__.__name__}, output"
                    )
                    msg_out = (
                        f"_transform output of {type(self)} does not comply "
                        "with sktime mtype specifications. See datatypes.MTYPE_REGISTER"
                        " for mtype specifications."
                    )
                    check_is_error_msg(
                        Xtd,
                        var_name=msg_start,
                        allowed_msg=msg_out,
                        raise_exception=True,
                    )

                if X_input_mtype == "pd.Series" and not metadata["is_univariate"]:
                    X_output_mtype = "pd.DataFrame"
            elif self.get_tag("scitype:transform-input") == "Panel":
                # Converting Panel to Series
                if X_input_scitype == "Hierarchical":
                    # Input was Hierarchical, but output has dropped one level.
                    # One level Hierarchical should be converted to Panel, but
                    # deeper Hierarchical should be converted to Hierarchical.
                    # Choose the simplest structure of the two.
                    X_output_mtype = ["pd-multiindex", "pd_multiindex_hier"]
                    output_scitype = ["Panel", "Hierarchical"]
                else:
                    # Input must have been Panel, output should be Series
                    X_output_mtype = "pd.DataFrame"
            else:
                # Input can be Panel or Hierarchical, since it is supported
                # by the used mtype
                output_scitype = X_input_scitype
                # Xt_mtype = metadata["mtype"]
            # else:
            #     Xt_mtype = X_input_mtype

            # Xt = convert(
            #     Xt,
            #     from_type=Xt_mtype,
            #     to_type=X_output_mtype,
            #     as_scitype=X_input_scitype,
            #     store=_converter_store_X,
            #     store_behaviour="freeze",
            # )
            return convert_to(
                Xt,
                to_type=X_output_mtype,
                as_scitype=output_scitype,
                store=_converter_store_X,
                store_behaviour="freeze",
            )
        elif output_scitype == "Primitives":
            # vectorization causes a superfluous zero level
            # if we have a Series input that is vectorized,
            # as in that case the index should be "0-level" (no levels)
            # but this is not possible in pandas, so it will have a level
            # which always has the entry 0.
            # in this case, we need to strip this level
            Xt_has_superfluous_zero_level = (
                X_input_scitype != "Series"
                and case == "case 3: requires vectorization"
                and isinstance(Xt, (pd.DataFrame, pd.Series))
            )
            # we ensure the output is pd_DataFrame_Table
            # & ensure the returned index is sensible
            # for return index, we need to deal with last level, constant 0
            if Xt_has_superfluous_zero_level:
                # if index is multiindex, last level is constant 0
                # and other levels are hierarchy
                if isinstance(Xt.index, pd.MultiIndex):
                    Xt.index = Xt.index.droplevel(-1)
                # we have an index with only zeroes, and should be reset to RangeIndex
                else:
                    Xt = Xt.reset_index(drop=True)
            return convert_to(
                Xt,
                to_type="pd_DataFrame_Table",
                as_scitype="Table",
                # no converter store since this is not a "1:1 back-conversion"
            )
        # else output_scitype is "Panel" and no need for conversion

        return Xt

    def _vectorize(self, methodname, **kwargs):
        """Vectorized/iterated loop over method of BaseTransformer.

        Uses transformers_ attribute to store one transformer per loop index.
        """
        X = kwargs.get("X")
        y = kwargs.pop("y", None)
        kwargs["args_rowvec"] = {"y": y}
        kwargs["rowname_default"] = "transformers"
        kwargs["colname_default"] = "transformers"

        FIT_METHODS = ["fit", "update"]
        TRAFO_METHODS = ["transform", "inverse_transform"]

        # fit-like methods: run method; clone first if fit
        if methodname in FIT_METHODS:
            if methodname == "fit":
                transformers_ = X.vectorize_est(
                    self,
                    method="clone",
                    rowname_default="transformers",
                    colname_default="transformers",
                    # no backend parallelization necessary for clone
                )
            else:
                transformers_ = self.transformers_

            self.transformers_ = X.vectorize_est(
                transformers_,
                method=methodname,
                backend=self.get_config()["backend:parallel"],
                backend_params=self.get_config()["backend:parallel:params"],
                **kwargs,
            )
            return self

        if methodname in TRAFO_METHODS:
            # loop through fitted transformers one-by-one, and transform series/panels
            if not self.get_tag("fit_is_empty"):
                # if not fit_is_empty: check index compatibility, get fitted trafos
                n_trafos = len(X)
                n, m = self.transformers_.shape
                n_fit = n * m
                if n_trafos != n_fit:
                    raise RuntimeError(
                        f"{type(self).__name__} is a transformer that applies per "
                        "individual time series, and broadcasts across instances. "
                        f"In fit, {type(self).__name__} makes one fit per instance, "
                        "and applies that fit to the instance with the same index in "
                        "transform. Vanilla use therefore requires the same number "
                        "of instances in fit and transform, but "
                        "found different number of instances in transform than in fit. "
                        f"number of instances seen in fit: {n_fit}; "
                        f"number of instances seen in transform: {n_trafos}. "
                        "For fit/transforming per instance, e.g., for pre-processing "
                        "in a time series classification, regression or clustering "
                        "pipeline, wrap this transformer in "
                        "FitInTransform, from sktime.transformations.compose."
                    )

                transformers_ = self.transformers_

            else:
                # if fit_is_empty: don't store transformers, run fit/transform in one
                transformers_ = X.vectorize_est(
                    self,
                    method="clone",
                    rowname_default="transformers",
                    colname_default="transformers",
                    # no backend parallelization necessary for clone
                )
                transformers_ = X.vectorize_est(
                    transformers_,
                    method="fit",
                    backend=self.get_config()["backend:parallel"],
                    backend_params=self.get_config()["backend:parallel:params"],
                    **kwargs,
                )

            # transform the i-th series/panel with the i-th stored transformer
            Xts = X.vectorize_est(
                transformers_,
                method=methodname,
                return_type="list",
                backend=self.get_config()["backend:parallel"],
                backend_params=self.get_config()["backend:parallel:params"],
                **kwargs,
            )
            Xt = X.reconstruct(Xts, overwrite_index=False)

            return Xt

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

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

        See extension_templates/transformer.py for implementation details.
        """
        # default fit is "no fitting happens"
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

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

        See extension_templates/transformer.py for implementation details.
        """
        raise NotImplementedError("abstract method")

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X, and conforming to mtype format specifications

        See extension_templates/transformer.py for implementation details.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, y=None):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _update must support all types in it
            Data to update transformer with
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator

        See extension_templates/transformer.py for implementation details.
        """
        # standard behaviour: no update takes place, new data is ignored
        return self


# initialize dynamic docstrings
BaseTransformer._init_dynamic_doc()


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

    # default tag values for "Panel-to-Tabular"
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

    # default tag values for "Panel-to-Panel"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
    }
