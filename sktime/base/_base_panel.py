# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Intermediate base class mixin with common functionality for panel tasks.

Inherits from BaseEstimator, descendents are:

BaseClassifier
BaseRegressor
"""

__author__ = ["fkiraly"]
__all__ = ["BasePanelMixin"]


import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.utils.warnings import warn


class BasePanelMixin(BaseEstimator):
    """Abstract base class for time series panel tasks, e.g., classifiers, regressors.

    The class contains boilerplate for checks and data conversions,
    which are common to multiple descendants, most importantly,
    BaseClassifier and BaseRegressor.
    """

    # convenience constant to control which metadata of input data
    # are regularly retrieved in input checks
    METADATA_REQ_IN_CHECKS = [
        "n_instances",
        "has_nans",
        "is_univariate",
        "is_equal_length",
    ]

    # attribute name where vectorized estimators are stored
    VECTORIZATION_ATTR = "estimators_"  # e.g., classifiers_, regressors_

    # used in error messages
    TASK = "panel data tasks"  # e.g., classification, regression
    EST_TYPE = "estimator"  # e.g., classifier, regressor
    EST_TYPE_PLURAL = "estimators"  # e.g., classifiers, regressors

    def _vectorize(self, methodname, **kwargs):
        """Vectorized/iterated loop over methods of BaseClassifier, BaseRegressor.

        Stores one estimator per loop index.
        """
        # retrieve data arguments
        X = kwargs.pop("X", None)
        y = kwargs.pop("y", None)

        # add some common arguments to kwargs
        kwargs["rowname_default"] = self.EST_TYPE_PLURAL
        kwargs["colname_default"] = self.EST_TYPE_PLURAL
        kwargs["backend"] = self.get_config()["backend:parallel"]
        kwargs["backend_params"] = self.get_config()["backend:parallel:params"]

        if methodname == "fit":
            self._yvec = y

            ests_ = self._yvec.vectorize_est(self, method="clone", **kwargs)
            ests_fit = self._yvec.vectorize_est(
                ests_,
                method=methodname,
                args={"y": y},
                X=X,
                **kwargs,
            )
            setattr(self, self.VECTORIZATION_ATTR, ests_fit)
            return self
        else:  # methodname == "predict" or methodname == "predict_proba":
            ests_ = getattr(self, self.VECTORIZATION_ATTR)
            y_preds = self._yvec.vectorize_est(
                ests_,
                method=methodname,
                X=X,
                return_type="list",
                **kwargs,
            )

            y_cols = self._y_metadata["feature_names"]

            if isinstance(y_preds[0], np.ndarray):
                y_preds_df = [pd.DataFrame(y_pred) for y_pred in y_preds]
                y_pred = pd.concat(y_preds_df, axis=1, keys=y_cols)
                if methodname == "predict":
                    # to avoid column MultiIndex with duplicated label
                    y_pred.columns = y_cols
            else:  # pd.DataFrame
                y_pred = pd.concat(y_preds, axis=1, keys=y_cols)

            return y_pred

    def _fit_predict_boilerplate(
        self,
        X,
        y,
        cv,
        change_state,
        method,
        return_type="single_y_pred",
    ):
        """Boilerplate logic for fit_predict and fit_predict_proba."""
        from sklearn.model_selection import KFold

        from sktime.datatypes import convert

        if isinstance(cv, int):
            random_state = getattr(self, "random_state", None)
            cv = KFold(cv, random_state=random_state, shuffle=True)

        if change_state:
            self.reset()
            est = self
        else:
            est = self.clone()

        if cv is None:
            return getattr(est.fit(X, y), method)(X)
        elif change_state:
            self.fit(X, y)

        # we now know that cv is an sklearn splitter
        X, y = self._internal_convert(X, y)
        X_metadata = self._check_input(
            X, y, return_metadata=self.METADATA_REQ_IN_CHECKS
        )
        X_mtype = X_metadata["mtype"]
        # Check this estimator can handle characteristics
        self._check_capabilities(X_metadata)

        # handle single class case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X)

        # Convert data to format easily usable for applying cv
        if isinstance(X, np.ndarray):
            X = convert(
                X,
                from_type=X_mtype,
                to_type="numpy3D",
                as_scitype="Panel",
                store_behaviour="freeze",
            )
        else:
            X = convert(
                X,
                from_type=X_mtype,
                to_type=["pd-multiindex", "nested_univ"],
                as_scitype="Panel",
                store_behaviour="freeze",
            )

        y_preds = []
        tt_ixx = []

        if isinstance(X.index, pd.MultiIndex):
            X_ix = X.index.get_level_values(0).unique()
        else:
            X_ix = np.arange(len(X))

        for tr_idx, tt_idx in cv.split(X_ix):
            X_train = self._subset(X, tr_idx)
            X_test = self._subset(X, tt_idx)
            y_train = self._subset(y, tr_idx)
            fitted_est = self.clone().fit(X_train, y_train)
            y_preds.append(getattr(fitted_est, method)(X_test))
            tt_ixx.append(tt_idx)

        if return_type == "single_y_pred":
            return self._pool(y_preds, tt_ixx, y)
        else:
            return y_preds

    def _subset(self, obj, ix):
        """Subset input data by ix, for use in fit_predict_boilerplate.

        Parameters
        ----------
        obj : pd.DataFrame or np.ndarray
            if pd.DataFrame, instance index = first level of pd.MultiIndex
            if np.ndarray, instance index = 0-th axis
        ix : sklearn splitter index, e.g., ix, _ from KFold.split(X)

        Returns
        -------
        obj_ix : obj subset by ix
        """
        if isinstance(obj, np.ndarray):
            return obj[ix]
        if not isinstance(obj, (pd.DataFrame, pd.Series)):
            raise ValueError("obj must be a pd.DataFrame, pd.Series, or np.ndarray")
        if not isinstance(obj.index, pd.MultiIndex):
            return obj.iloc[ix]
        else:
            ix_loc = obj.index.get_level_values(0).unique()[ix]
            return obj.loc[ix_loc]

    def _pool(self, y_preds, tt_ixx, y):
        """Pool predictions from cv splits, for use in fit_predict_boilerplate.

        Parameters
        ----------
        y_preds : list of np.ndarray or pd.DataFrame
            list of predictions from cv splits
        tt_ixx : list of np.ndarray or pd.DataFrame
            list of test indices from cv splits

        Returns
        -------
        y_pred : np.ndarray, pooled predictions
        """
        y_pred = y_preds[0]
        if isinstance(y_pred, (pd.DataFrame, pd.Series)):
            for i in range(1, len(y_preds)):
                y_pred = y_pred.combine_first(y_preds[i])
            y_pred = y_pred.reindex(y.index).fillna(-1)
        else:
            if y_pred.ndim == 1:
                sh = y.shape
            else:
                sh = (y.shape[0], y_pred.shape[1])
            y_pred = -np.ones(sh, dtype=y.dtype)
            for i, ix in enumerate(tt_ixx):
                y_preds_i = y_preds[i]
                if y_pred.ndim == 1:
                    y_preds_i = y_preds_i.reshape(-1)
                y_pred[ix] = y_preds_i
        return y_pred

    def _check_convert_X_for_predict(self, X):
        """Input checks, capability checks, repeated in all predict/score methods.

        Parameters
        ----------
        X : any object (to check/convert)
            should be of a supported Panel mtype or 2D numpy.ndarray

        Returns
        -------
        X: an object of a supported Panel mtype, numpy3D if X was a 2D numpy.ndarray

        Raises
        ------
        ValueError if X is of invalid input data type, or there is not enough data
        ValueError if the capabilities in self._tags do not handle the data.
        """
        X = self._internal_convert(X)
        X_metadata = self._check_input(X, return_metadata=self.METADATA_REQ_IN_CHECKS)
        X_mtype = X_metadata["mtype"]
        # Check that estimator can handle characteristics
        self._check_capabilities(X_metadata)
        # Convert data as dictated by the estimator tags
        X = self._convert_X(X, X_mtype=X_mtype)

        return X

    def _check_capabilities(self, X_metadata):
        """Check whether this estimator can handle the data characteristics.

        Parameters
        ----------
        missing : boolean, does the data passed to fit contain missing values?
        multivariate : boolean, does the data passed to fit contain missing values?
        unequal : boolea, do the time series passed to fit have variable lengths?

        Raises
        ------
        ValueError if the capabilities in self._tags do not handle the data.
        """
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]

        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")

        self_name = type(self).__name__

        # identify problems, mismatch of capability and inputs
        problems = []
        if missing and not allow_missing:
            problems += ["missing values"]
        if multivariate and not allow_multivariate:
            problems += ["multivariate series"]
        if unequal and not allow_unequal:
            problems += ["unequal length series"]

        # construct error message
        problems_and = " and ".join(problems)
        problems_or = " or ".join(problems)
        msg = (
            f"Data seen by {self_name} instance has {problems_and}, "
            f"but this {self_name} instance cannot handle {problems_or}. "
            f"Calls with {problems_or} may result in error or unreliable results."
        )

        # raise exception or warning with message
        # if self is composite, raise a warning, since passing could be fine
        #   see discussion in PR 2366 why
        if len(problems) > 0:
            if self.is_composite():
                warn(msg, obj=self)
            else:
                raise ValueError(msg)

    def _convert_X(self, X, X_mtype):
        """Convert equal length series from DataFrame to numpy array or vice versa.

        Parameters
        ----------
        X : input data for the estimator, any Panel mtype
        X_mtype : str, a Panel mtype string, e.g., "pd_multiindex", "numpy3D"

        Returns
        -------
        X : input X converted to type in "X_inner_mtype" tag
            usually a pd.DataFrame (nested) or 3D np.ndarray
            Checked and possibly converted input data
        """
        from sktime.datatypes import convert

        inner_type = self.get_tag("X_inner_mtype")
        # convert pd.DataFrame
        X = convert(
            X,
            from_type=X_mtype,
            to_type=inner_type,
            as_scitype="Panel",
        )
        return X

    def _check_y(self, y=None, return_to_mtype=False):
        """Check and coerce X/y for fit/transform functions.

        Parameters
        ----------
        y : pd.DataFrame, pd.Series or np.ndarray
        return_to_mtype : bool
            whether to return the mtype of y output

        Returns
        -------
        y_inner : object of sktime compatible time series type
            can be Series, Panel, Hierarchical
        y_metadata : dict
            metadata of y, returned by check_is_scitype
        y_mtype : str, only returned if return_to_mtype=True
            mtype of y_inner, after convert
        """
        from sktime.datatypes import (
            MTYPE_LIST_TABLE,
            VectorizedDF,
            check_is_error_msg,
            check_is_scitype,
            convert,
        )
        from sktime.datatypes._dtypekind import DtypeKind

        if y is None:
            if return_to_mtype:
                return None, None, None
            else:
                return None, None

        capa_multioutput = self.get_tag("capability:multioutput")
        y_inner_mtype = self.get_tag("y_inner_mtype")

        y_metadata_required = ["is_univariate", "feature_names", "feature_kind"]
        y_valid, y_msg, y_metadata = check_is_scitype(
            y, "Table", return_metadata=y_metadata_required
        )

        if not y_valid:
            allowed_msg = (
                f"In {self.TASK}, y must be of a supported type, "
                f"for instance 1D or 2D numpy arrays, pd.DataFrame, or pd.Series. "
                f"Allowed compatible mtype format specifications are:"
                f" {MTYPE_LIST_TABLE} ."
            )
            check_is_error_msg(
                y_msg, var_name="y", allowed_msg=allowed_msg, raise_exception=True
            )

        est_type = self.get_tag("object_type")  # classifier or regressor
        if (
            est_type == "regressor"
            and DtypeKind.CATEGORICAL in y_metadata["feature_kind"]
        ):
            raise TypeError(
                "Regressors do not support categorical features in endogeneous y."
            )

        y_uni = y_metadata["is_univariate"]
        y_mtype = y_metadata["mtype"]

        requires_vectorization = not capa_multioutput and not y_uni

        if requires_vectorization:
            y_df = convert(
                y,
                from_type=y_mtype,
                to_type="pd_DataFrame_Table",
                as_scitype="Table",
                store=self._converter_store_y,
            )
            y_vec = VectorizedDF([y_df], iterate_cols=True)
            if return_to_mtype:
                return y_vec, y_metadata, "pd_DataFrame_Table"
            else:
                return y_vec, y_metadata

        y_inner, y_inner_mtype = convert(
            y,
            from_type=y_mtype,
            to_type=y_inner_mtype,
            as_scitype="Table",
            store=self._converter_store_y,
            return_to_mtype=True,
        )

        if return_to_mtype:
            return y_inner, y_metadata, y_inner_mtype
        else:
            return y_inner, y_metadata

    def _get_output_mtype(self, y):
        """Get the mtype of the output y.

        Parameters
        ----------
        y : np.ndarray or pd.DataFrame
            output to convert

        Returns
        -------
        y_mtype : str
            mtype of y
        """
        from sktime.datatypes import check_is_scitype

        y_mtype = check_is_scitype(y, "Table", return_metadata="mtype")
        return y_mtype

    def _convert_output_y(self, y):
        """Convert output y to original format.

        Parameters
        ----------
        y : np.ndarray or pd.DataFrame
            output to convert

        Returns
        -------
        y : np.ndarray or pd.DataFrame
        """
        from sktime.datatypes import convert

        # for consistency with legacy behaviour:
        # output is coerced to numpy1D in case of univariate output
        if not self._y_metadata["is_univariate"]:
            output_mtype = self._y_metadata["mtype"]
            converter_store = self._converter_store_y
        else:
            output_mtype = "numpy1D"
            converter_store = None

        # inner return mtype is what we convert from
        # special treatment for 1D numpy array
        # this can be returned in composites due to
        # current downwards compatible choice "1D return is always numpy"
        if isinstance(y, np.ndarray) and y.ndim == 1:
            inner_return_mtype = "numpy1D"
        else:
            inner_return_mtype = self._y_inner_mtype

        y = convert(
            y,
            from_type=inner_return_mtype,
            to_type=output_mtype,
            as_scitype="Table",
            store=converter_store,
            store_behaviour="freeze",
        )
        return y

    def _check_input(self, X, y=None, enforce_min_instances=1, return_metadata=True):
        """Check whether input X and y are valid formats with minimum data.

        Raises a ValueError if the input is not valid.

        Parameters
        ----------
        X : check whether conformant with any sktime Panel mtype specification
        y : check whether a pd.Series or np.array
        enforce_min_instances : int, optional (default=1)
            check there are a minimum number of instances.
        return_metadata : bool, str, or list of str
            metadata fields to return with X_metadata, input to check_is_scitype

        Returns
        -------
        metadata : dict with metadata for X returned by datatypes.check_is_scitype

        Raises
        ------
        ValueError
            If y or X is invalid input data type, or there is not enough data
        """
        from sktime.datatypes import (
            MTYPE_LIST_PANEL,
            check_is_error_msg,
            check_is_scitype,
        )
        from sktime.datatypes._dtypekind import DtypeKind

        # Check X is valid input type and recover the data characteristics
        X_valid, msg, X_metadata = check_is_scitype(
            X, scitype="Panel", return_metadata=return_metadata
        )

        # raise informative error message if X is in wrong format
        allowed_msg = (
            f"Allowed scitypes for {self.EST_TYPE_PLURAL} are Panel mtypes, "
            f"for instance a pandas.DataFrame with MultiIndex and last(-1) "
            f"level an sktime compatible time index. "
            f"Allowed compatible mtype format specifications are: {MTYPE_LIST_PANEL} ."
        )
        if not X_valid:
            check_is_error_msg(
                msg, var_name="X", allowed_msg=allowed_msg, raise_exception=True
            )

        est_type = self.get_tag("object_type")  # classifier or regressor
        if DtypeKind.CATEGORICAL in X_metadata["feature_kind"] and not self.get_tag(
            "capability:categorical_in_X"
        ):
            raise TypeError(
                f"{est_type}s do not support categorical features in exogeneous X."
            )

        n_cases = X_metadata["n_instances"]
        if n_cases < enforce_min_instances:
            raise ValueError(
                f"Minimum number of cases required is {enforce_min_instances} but X "
                f"has : {n_cases}"
            )

        # Check y if passed
        if y is not None:
            # Check y valid input
            if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
                raise ValueError(
                    "y must be a np.array or a pd.Series or pd.DataFrame, but found ",
                    f"type: {type(y)}",
                )
            # Check matching number of labels
            n_labels = y.shape[0]
            if n_cases != n_labels:
                raise ValueError(
                    f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                    f"{n_labels}"
                )
            if isinstance(y, np.ndarray):
                if y.ndim > 2:
                    raise ValueError(
                        f"np.ndarray y must be 1-dimensional or 2-dimensional, "
                        f"but found {y.ndim} dimensions"
                    )
            # warn if only a single class label is seen
            # this should not raise exception since this can occur by train subsampling
            if len(np.unique(y)) == 1:
                warn(
                    "only single label seen in y passed to "
                    f"fit of {self.EST_TYPE} {type(self).__name__}",
                    obj=self,
                )

        return X_metadata

    def _internal_convert(self, X, y=None):
        """Convert X and y if necessary as a user convenience.

        Convert X to a 3D numpy array if already a 2D and convert y into an 1D numpy
        array if passed as a Series.

        Parameters
        ----------
        X : an object of a supported Panel mtype, or 2D numpy.ndarray
        y : np.ndarray or pd.Series

        Returns
        -------
        X: an object of a supported Panel mtype, numpy3D if X was a 2D numpy.ndarray
        y: np.ndarray
        """
        if isinstance(X, np.ndarray):
            # Temporary fix to insist on 3D numpy. For univariate problems,
            # most panel estimators simply convert back to 2D. This squeezing should be
            # done here, but touches a lot of files, so will get this to work first.
            if X.ndim == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
        if y is None:
            return X
        return X, y
