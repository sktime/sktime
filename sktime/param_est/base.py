# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for parameter estimator scitype.

    class name: BaseParamFitter

Scitype defining methods:
    fitting                - fit(X, y)
    updating               - update(y, X=None)
    get fitted parameters  - get_fitted_params() -> dict

Inspection methods:
    hyper-parameter inspection  - get_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["fkiraly", "satvshr"]

__all__ = ["BaseParamFitter"]

from sktime.base import BaseEstimator
from sktime.datatypes import (
    check_is_scitype,
    convert,
    scitype_to_mtype,
    update_data,
)
from sktime.datatypes._dtypekind import DtypeKind
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.warnings import warn


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseParamFitter(BaseEstimator):
    """Base parameter fitting estimator class.

    This base class defines the interface for parameter estimators that accept
    both feature data (X) and target values (y). It provides methods for fitting,
    updating, and inspecting the fitted parameters.

    Parameters
    ----------
    None

    Attributes
    ----------
    _is_fitted : bool
        Flag indicating whether the estimator has been fitted.
    _X : sktime compatible container
        Stored feature data from fit and update.
    _y : array-like
        Stored target data from fit and update.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "param_est",  # type of object
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which types do _fit/_predict support for y?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "scitype:y": "univariate",  # which y are fine: univariate/multivariate/both
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "capability:contains_y": False,  # does estimator require y?
        "capability:pairwise": False,  # can handle pairwise parameter estimation?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # string or str list of pkg soft dependencies
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
    }

    def __init__(self):
        self._is_fitted = False

        self._X = None
        self._y = None

        super().__init__()
        _check_estimator_deps(self)

    def __mul__(self, other):
        """Magic * method, for estimators on the right.

        Overloaded multiplication operation for parameter fitters.
        Implemented for ``other`` being:

        * a forecaster, results in ``PluginParamsForecaster``
        * a transformer, results in ``PluginParamsTransformer``
        * otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` estimator, must be one of the types specified above
            otherwise, `NotImplemented` is returned

        Returns
        -------
        one of the plugin estimator objects,
        concatenation of `self` (first) with `other` (last).
        """
        from sktime.forecasting.base import BaseForecaster
        from sktime.param_est.plugin import (
            PluginParamsForecaster,
            PluginParamsTransformer,
        )
        from sktime.transformations.base import BaseTransformer

        if isinstance(other, BaseForecaster):
            return PluginParamsForecaster(param_est=self, forecaster=other)
        elif isinstance(other, BaseTransformer):
            return PluginParamsTransformer(param_est=self, transformer=other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Magic * method, return concatenated ParamFitterPipeline, trafos on left.

        Overloaded multiplication operation for parameter fitters.
        Implemented for ``other`` being a transformer,
        otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        BaseParamFitter object, concatenation of `other` (first) with `self` (last).
        """
        from sktime.param_est.compose import ParamFitterPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        #  todo: this can probably be simplified further with "zero length" pipelines
        if isinstance(other, BaseTransformer):
            # ClassifierPipeline already has the dunder method defined
            if isinstance(self, ParamFitterPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return ParamFitterPipeline(param_est=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a ClassifierPipeline
            else:
                return ParamFitterPipeline(param_est=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def fit(self, X, y=None):
        """Fit estimator and estimate parameters.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes `X` to self._X.
            Writes `y` to self._y if y is not None.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : sktime-compatible container
            Time series data (features) in a format supported by the estimator.
        y : array-like, optional (default=None)
            Target values for supervised parameter estimation. If the estimator does not
            require target values (i.e., its tag "capability:contains_y" is False),
            y must be None.

        Returns
        -------
        self : Reference to self.
        """
        # check X is not None
        assert X is not None, "X cannot be None, but found None"

        # if fit is called, estimator is reset, including fitted state
        self.reset()

        # Check and convert X and y according to estimator requirements
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X to the new X
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################
        self._fit(X=X_inner, y=y_inner)

        # this should happen last
        self._is_fitted = True

        return self

    def update(self, X, y=None):
        """Update fitted parameters on more data.

        This method allows for updating the estimator with additional observations.
        If a custom update method is not implemented, the default behavior is to refit
        the estimator using all data seen so far. Note that the primary parameter is y;
        if X is provided, it is used to update the stored feature data.

        Parameters
        ----------
        X : sktime-compatible container, optional (default=None)
            New time series data (features) for updating the estimator. If not provided,
            only y is updated.
        y : array-like
            New target values for updating the estimator. An empty or None y will result
            in no update and a warning will be issued.

        Returns
        -------
        self : reference to self
        """
        self.check_is_fitted()

        if y is None or (hasattr(y, "__len__") and len(y) == 0):
            warn(
                f"empty y passed to update of {self}, no update was carried out",
                obj=self,
            )
            return self

        if X is not None:
            # Validate and convert the new X and y data
            X_inner, y_inner = self._check_X_y(X=X, y=y)
            self._update_y_X(y_inner, X_inner)
        else:
            # Update only the target data
            self._update_y_X(y, None)

        # Pass the checked and converted data to the estimator-specific _update logic
        self._update(X=X_inner, y=y_inner)

        return self

    def _check_X_y(self, X=None, y=None):
        """Validate and coerce feature and target data for fit and update.

        This method performs several checks and conversions:
          - If both X and y are None, returns (None, None).
          - If y is provided but the estimator's "capability:contains_y" tag is False,
            a ValueError is raised.
          - For y:
              * Checks that y is in one of the allowed sktime formats for its scitype.
              * Ensures y does not contain categorical values.
              * For multivariate estimators, ensures y has multiple variables.
              * Checks for missing values if the estimator does not support them.
              * Converts y to the format specified by the "y_inner_mtype" tag.
          - For X:
              * If the estimator is pairwise (tag "capability:pairwise" is True), X must
                 be a square pairwise distance/similarity matrix; y is ignored.
              * Otherwise, checks that X is in one of the allowed sktime formats for
              its declared scitype.
              * Ensures X does not contain categorical values.
              * Checks for missing values if not supported.
              * Converts X to the format specified by the "X_inner_mtype" tag.
          - If both X and y are provided (and the estimator is not pairwise), verifies
            that the lengths of X and y match.

        Parameters
        ----------
        X : sktime-compatible container, optional (default=None)
            Feature data (time series) to be validated and converted.
        y : array-like, optional (default=None)
            Target values for parameter estimation to be validated and converted.

        Returns
        -------
        X_inner : object or None
            The converted feature data, in the format specified by the estimator's tag
            "X_inner_mtype". Returns None if X is None.
        y_inner : object or None
            The converted target data, in the format specified by the estimator's tag
            "y_inner_mtype". Returns None if y is None or if the estimator is pairwise.

        Raises
        ------
        TypeError
            If X or y is not in an allowed sktime-compatible format, or if y contains
            categorical values.
        ValueError
            If:
              * y is provided for an estimator that does not support target values.
              * For multivariate y, if only univariate data is provided when
              multivariate is required.
              * For pairwise estimators, if X is not square.
              * The lengths of X and y do not match.
        """
        X_inner, y_inner = None, None
        if X is None and y is None:
            return None, None

        if y is not None and self.get_tag("capability:contains_y") is False:
            raise ValueError(
                f"{type(self).__name__} does not require y, but y was passed."
            )

        def _check_missing(obj):
            """Raise an error if obj contains missing values (if not supported)."""
            if obj.isnull().any() and not self.get_tag("capability:missing_values"):
                msg = (
                    f"{type(self).__name__} cannot handle missing data (nans), "
                    f"but the passed data contained missing values."
                )
                raise ValueError(msg)

        # Process y if provided
        if y is not None:
            ALLOWED_SCITYPES = [_coerce_to_list(self.get_tag("scitype:y"))]
            FORBIDDEN_MTYPES = []

            # Prepare a message about allowed mtypes
            for scitype in ALLOWED_SCITYPES:
                mtypes = set(scitype_to_mtype(scitype))
                mtypes = list(mtypes.difference(FORBIDDEN_MTYPES))
                mtypes_msg = f'"For {scitype} scitype: {mtypes}. '

            y_valid, _, y_metadata = check_is_scitype(
                y,
                scitype=ALLOWED_SCITYPES,
                return_metadata=["feature_kind"],
                var_name="y",
            )
            msg = (
                "y must be in an sktime compatible format, "
                f"of scitypes {ALLOWED_SCITYPES}, for example a pandas.DataFrame with "
                "an sktime compatible time index. See the data format tutorial for "
                "more details. "
            )
            if not y_valid:
                raise TypeError(msg + mtypes_msg)

            if DtypeKind.CATEGORICAL in y_metadata["feature_kind"]:
                raise TypeError(
                    "Parameter estimators do not support categorical features in y."
                )

            if (
                self.get_tag("scitype:y") == "multivariate"
                and y_metadata["is_univariate"]
            ):
                raise ValueError(
                    f"Unsupported input data type in {type(self).__name__}: "
                    "this estimator accepts only strictly multivariate data. "
                    "y must have two or more variables, but only one was found."
                )

            _check_missing(y)
            y_mtype = y_metadata["mtype"]
            y_inner_mtype = _coerce_to_list(self.get_tag("y_inner_mtype"))
            y_scitype = y_metadata["scitype"]
            y_inner = convert(
                y,
                from_type=y_mtype,
                to_type=y_inner_mtype,
                as_scitype=y_scitype,
            )

        if X is not None:
            # Handle pairwise estimators separately.
            if self.get_tag("capability:pairwise"):
                ALLOWED_MTYPES = ["pd.DataFrame", "numpy2D", "numpy3D"]

                X_valid, _, X_metadata = check_is_scitype(
                    X,
                    scitype=["Pairwise"],
                    return_metadata=["n_instances", "n_features"],
                )

                if not X_valid:
                    raise TypeError(
                        f"X must be a valid pairwise matrix (distance/similarity), "
                        f"expected one of {ALLOWED_MTYPES} but got {type(X)}."
                    )

                if X_metadata["n_instances"] != X_metadata["n_features"]:
                    raise ValueError(
                        "Pairwise matrix X must be square (n_samples x n_samples)."
                    )

                return X, None  # For pairwise estimators, y is not used.

            else:
                ALLOWED_SCITYPES = _coerce_to_list(self.get_tag("scitype:X"))
                FORBIDDEN_MTYPES = ["numpyflat", "pd-wide"]

                for scitype in ALLOWED_SCITYPES:
                    mtypes = set(scitype_to_mtype(scitype))
                    mtypes = list(mtypes.difference(FORBIDDEN_MTYPES))
                    mtypes_msg = f'"For {scitype} scitype: {mtypes}. '

                X_valid, _, X_metadata = check_is_scitype(
                    X,
                    scitype=ALLOWED_SCITYPES,
                    return_metadata=["feature_kind"],
                    var_name="X",
                )
                msg = (
                    "X must be in an sktime compatible format, "
                    f"of scitypes {ALLOWED_SCITYPES}, for example a pandas.DataFrame "
                    "sktime compatible time index. "
                )
                if not X_valid:
                    raise TypeError(msg + mtypes_msg)

                if DtypeKind.CATEGORICAL in X_metadata["feature_kind"]:
                    raise TypeError(
                        "Parameter estimators do not support categorical features in X."
                    )

            _check_missing(X)
            X_mtype = X_metadata["mtype"]
            X_inner_mtype = _coerce_to_list(self.get_tag("X_inner_mtype"))
            X_scitype = X_metadata["scitype"]
            X_inner = convert(
                X,
                from_type=X_mtype,
                to_type=X_inner_mtype,
                as_scitype=X_scitype,
            )

        # If both X and y are provided, check that their lengths match.
        if y_inner is not None and len(y_inner) != len(X_inner):
            raise ValueError(
                f"Length of X ({len(X_inner)}) and y ({len(y_inner)}) must match."
            )

        return X_inner, y_inner

    def _check_X(self, X=None):
        """Validate and coerce feature data X.

        This is a convenience method that calls _check_X_y with only X provided.

        Parameters
        ----------
        X : sktime-compatible container, optional (default=None)
            Feature data to be validated and converted.

        Returns
        -------
        X_inner : object or None
            The converted feature data in the format specified by the estimator.
        """
        return self._check_X_y(X=X)[0]

    def _update_y_X(self, y, X):
        """Update internal memory with the provided target and feature data.

        If the estimator's configuration indicates that data should be remembered
        (see config "remember_data"), the new data will be appended to the stored
        `_y` and `_X` attributes.

        Parameters
        ----------
        y : array-like or None
            New target data to be remembered. If None, `_y` remains unchanged.
        X : sktime-compatible container or None
            New feature data to be remembered. If None, `_X` remains unchanged.
        """
        if y is not None and self.get_config()["remember_data"]:
            if not hasattr(self, "_y") or self._y is None or not self.is_fitted:
                self._y = y
            else:
                self._y = update_data(self._y, y)

        if X is not None:
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            else:
                self._X = update_data(self._X, X)

    def _update_X(self, X):
        """Update internal memory with new feature data X.

        Parameters
        ----------
        X : sktime-compatible container
            New feature data to be remembered.
        """
        self._update_y_X(None, X)

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.
        y : array-like, optional (default=None)
            Target values in the format specified by the estimators "y_inner_mtype" tag.

        Returns
        -------
        self : BaseParamFitter
            Reference to the estimator with updated fitted attributes.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, y):
        """Update fitted parameters on more data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Writes to self:
            Sets fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series with which to update the estimator.
        y : array-like, optional (default=None)
            New target data in the format specified by "y_inner_mtype".

        Returns
        -------
        self : BaseParamFitter
            Reference to the updated estimator.
        """
        warn(
            f"NotImplementedWarning: {self.__class__.__name__} "
            f"does not have a custom `update` method implemented. "
            f"{self.__class__.__name__} will be refit each time "
            f"`update` is called.",
            obj=self,
        )
        # Default behavior: refit with all stored data
        self.fit(X=self._X, y=self._y)
        # todo: should probably be self._fit, not self.fit
        # but looping to self.fit for now to avoid interface break

        return self
