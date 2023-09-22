#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adaptor for applying Scikit-learn-like transformers to time series."""

__author__ = ["mloning", "fkiraly"]
__all__ = ["TabularToSeriesAdaptor"]

from inspect import signature

import numpy as np
from sklearn.base import clone

from sktime.transformations.base import BaseTransformer


class TabularToSeriesAdaptor(BaseTransformer):
    """Adapt scikit-learn transformation interface to time series setting.

    This is useful for applying scikit-learn :term:`tabular` transformations
    to :term:`series <Time series>`, but only works with transformations that
    do not require multiple :term:`instances <instance>` for fitting.

    The adaptor behaves as follows.
    If fit_in_transform = False and X is a series (pd.DataFrame, pd.Series, np.ndarray):
        ``fit(X)`` fits a clone of ``transformer`` to X (considered as a table)
        ``transform(X)`` applies transformer.transform to X and returns the result
        ``inverse_transform(X)`` applies transformer.inverse_transform to X
    If fit_in_transform = True and X is a series (pd.DataFrame, pd.Series, np.ndarray):
        ``fit`` is empty
        ``transform(X)`` applies transformer.fit(X).transform(X) to X,
            considered as a table, and returns the result
        ``inverse_transform(X)`` applies transformer.fit(X).inverse_transform(X) to X

    If fit_in_transform = False, and X is of a panel/hierarchical type:
        ``fit(X)`` fits a clone of ``transformer`` for each individual series x in X
        ``transform(X)`` applies transform(x) of the clone belonging to x,
                (where the index of x in transform equals the index of x in fit)
            for each individual series x in X, and returns the result
        ``inverse_transform(X)`` applies transform(x) of the clone belonging to x,
                (where the index of x in transform equals the index of x in fit)
            for each individual series x in X, and returns the result
        Note: instances indices in transform/inverse_transform
            must be equal to those seen in fit
    If fit_in_transform = True, and X is of a panel/hierarchical type:
        ``fit`` is empty
        ``transform(X)`` applies transformer.fit(x).transform(x)
            to all individual series x in X and returns the result
        ``inverse_transform(X)`` applies transformer.fit(x).inverse_transform(x)
            to all individual series x in X and returns the result

    WARNING: if fit_in_transform is set to False,
        when applied to Panel or Hierarchical data,
        the resulting transformer will identify individual series in test set
        with series indices in training set, on which instances were fit
        in particular, transform will not work if number of instances
            and indices of instances in transform are different from those in fit
    WARNING: if fit_in_transform is set to True,
        then each series in the test set will be transformed as batch by fit-predict,
        this may cause information leakage in a forecasting setting
            (but not in a time series classification/regression/clustering setting,
            because in these settings the independent samples are the individual series)

    Whether ``y`` is passed to transformer methods is controlled by ``pass_y``.
    If the inner transformer has non-defaulting ``y`` args, the default behaviour is
    to pass ``y`` to ``fit``, ``fit_transform``, or ``transform``.
    If no ``y`` arg is present, or if it has a default value, ``y`` is not passed.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like transformer to fit and apply to series.
        This is used as a "blueprint" and not fitted or otherwise mutated.
    fit_in_transform: bool, optional, default=False
        whether transformer_ should be fitted in transform (True), or in fit (False)
            recommended setting in forecasting (single series or hierarchical): False
            recommended setting in ts classification, regression, clustering: True
    pass_y : str, optional, one of "auto" (default), "fit", "always", "never"
        Whether to pass y to transformer methods of the ``transformer`` clone.
        "auto": passes y to methods fit, transform, fit_transform, inverse_transform,
            if and only if y is a named arg of either method without default.
            Note: passes y even if it is None
        "fit": passes y to method fit, but not to transform.
            Note: passes y even if it is None, or if not a named arg
        "always": passes y to all methods, fit, transform, inverse_transform.
            Note: passes y even if it is None, or if not a named arg
        "never": never passes y to any method.

    Attributes
    ----------
    transformer_ : Estimator
        Transformer that is fitted to data, clone of transformer.

    Examples
    --------
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = TabularToSeriesAdaptor(MinMaxScaler())
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "transform-returns-same-time-index": True,
        "fit_is_empty": False,
    }

    def __init__(self, transformer, fit_in_transform=False, pass_y="auto"):
        self.transformer = transformer
        self.transformer_ = clone(self.transformer)
        self.fit_in_transform = fit_in_transform
        self.pass_y = pass_y

        super().__init__()

        if hasattr(transformer, "inverse_transform"):
            self.set_tags(**{"capability:inverse_transform": True})

        # sklearn transformers that are known to fit in transform do not need fit
        if hasattr(transformer, "_get_tags"):
            trafo_fit_in_transform = transformer._get_tags()["stateless"]
        else:
            trafo_fit_in_transform = False

        self._skip_fit = fit_in_transform or trafo_fit_in_transform

        if self._skip_fit:
            self.set_tags(**{"fit_is_empty": True})

        trafo_has_y, trafo_has_y_default = self._trafo_has_y_and_default("fit")
        need_y = trafo_has_y and not trafo_has_y_default
        if need_y or pass_y not in ["auto", "no"]:
            self.set_tags(**{"y_inner_mtype": "numpy1D"})

    def _trafo_has_y_and_default(self, method="fit"):
        """Return if transformer.method has a y, and whether y has a default."""
        method_fun = getattr(self.transformer, method)
        method_params = list(signature(method_fun).parameters.keys())
        if "y" in method_params:
            y_param = signature(self.transformer.fit).parameters["y"]
            y_default = y_param.default
            y_has_default = y_default is not y_param.empty
            return True, y_has_default
        else:
            return False, False

    def _get_y_args(self, y, method="fit"):
        """Get empty dict or dict with y, depending on pass_y and method.

        The return is a dict which is passed to the method of name method,
        according to the pass_y setting.
        """
        pass_y = self.pass_y

        if pass_y == "auto":
            has_y, has_y_default = self._trafo_has_y_and_default(method)
            need_y = has_y and not has_y_default
            return_y = need_y
        elif pass_y == "fit":
            return_y = method in ["fit", "fit_transform"]
        elif pass_y == "always":
            return_y = True
        elif pass_y == "never":
            return_y = False
        else:
            raise ValueError(
                f"error in {self.__class__.__name__}, pass_y={pass_y} not supported, "
                "must be one of 'auto', 'fit', 'always', 'never'"
            )

        if return_y:
            return {"y": y}
        else:
            return {}

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : 2D np.ndarray
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        y_args = self._get_y_args(y, method="fit")

        if not self._skip_fit:
            self.transformer_.fit(X, **y_args)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        y_fit_args = self._get_y_args(y, method="fit")
        y_trafo_args = self._get_y_args(y, method="transform")

        if self._skip_fit:
            Xt = self.transformer_.fit(X, **y_fit_args).transform(X, **y_trafo_args)
        else:
            Xt = self.transformer_.transform(X)

        # coerce sensibly to 2D np.ndarray
        if isinstance(Xt, (int, float, str)):
            Xt = np.array([[Xt]])
        if not isinstance(Xt, np.ndarray):
            Xt = np.array(Xt)
        if Xt.ndim == 1:
            Xt = Xt.reshape((len(X), 1))

        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be inverse transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        y_fit_args = self._get_y_args(y, method="fit")
        y_i_args = self._get_y_args(y, method="inverse_transform")

        if self.fit_in_transform:
            Xt = self.transformer_.fit(X, **y_fit_args).inverse_transform(X, **y_i_args)
        else:
            Xt = self.transformer_.inverse_transform(X, **y_i_args)
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import StandardScaler

        params1 = {"transformer": StandardScaler(), "fit_in_transform": False}
        params2 = {
            "transformer": StandardScaler(),
            "fit_in_transform": True,
            "pass_y": "auto",
        }
        params3 = {"transformer": VarianceThreshold(), "pass_y": "fit"}
        params4 = {"transformer": VarianceThreshold()}

        return [params1, params2, params3, params4]


class PandasTransformAdaptor(BaseTransformer):
    """Adapt pandas transformations to sktime interface.

    In `transform`, executes `pd.DataFrame` method of name `method` on data,
    optionally with keywords arguments passed, via `kwargs` hyper-parameter.
    The `apply_to` parameter controls what the data is upon which `method` is called:
    "call" = for `X` seen in `transform`, "all"/"all_subset" = all data seen so far.
    See below for details.

    For hierarchical series, operation is applied by instance.

    Parameters
    ----------
    method : str, optional, default = None = identity transform
        name of the method of DataFrame that is applied in transform
    kwargs : dict, optional, default = empty dict (no kwargs passed to method)
        arguments passed to DataFrame.method
    apply_to : str, one of "call", "all", "all_subset", optional, default = "call"
        "call" = method is applied to `X` seen in transform only
        "all" = method is applied to all `X` seen in `fit`, `update`, `transform`
            more precisely, the application to `self._X` is returned
        "all_subset" = method is applied to all `X` like for "all" value,
            but before returning, result is sub-set to indices of `X` in `transform`
        in "all", "all_subset", `X` seen in `transform` do not update `self._X`

    Examples
    --------
    >>> from sktime.transformations.series.adapt import PandasTransformAdaptor
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()

    >>> transformer = PandasTransformAdaptor("diff")
    >>> y_hat = transformer.fit_transform(y)

    >>> transformer = PandasTransformAdaptor("diff", apply_to="all_subset")
    >>> y_hat = transformer.fit(y.iloc[:12])
    >>> y_hat = transformer.transform(y.iloc[12:])
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "transform-returns-same-time-index": False,
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "remember_data": False,  # remember all data seen as _X
    }

    def __init__(self, method, kwargs=None, apply_to="call"):
        self.method = method
        self.kwargs = kwargs
        self.apply_to = apply_to

        if not isinstance(apply_to, str):
            raise TypeError(
                f"apply_to parameter must be a str, but found {type(apply_to)}"
            )
        if apply_to not in ["call", "all", "all_subset"]:
            raise ValueError(
                'apply_to must be one of "call", "all", "all_subset", '
                f'but found "{apply_to}"'
            )

        super().__init__()

        if apply_to in ["all", "all_subset"]:
            self.set_tags(**{"remember_data": True})

        if apply_to == "all_subset":
            self.set_tags(**{"transform-returns-same-time-index": True})

        if apply_to == "call":
            self.set_tags(**{"fit_is_empty": True})

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame
            transformed version of X
        """
        apply_to = self.apply_to
        method = self.method
        kwargs = self.kwargs

        if kwargs is None:
            kwargs = {}

        if apply_to in ["all", "all_subset"]:
            _X = X.combine_first(self._X)
        else:
            _X = X

        Xt = getattr(_X, method)(**kwargs)

        if apply_to in ["all_subset"]:
            Xt = Xt.loc[X.index]

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"method": "diff"}
        params2 = {"method": "diff", "kwargs": {"periods": 2}, "apply_to": "all_subset"}
        params3 = {
            "method": "shift",
            "kwargs": {"periods": 12},
            "apply_to": "all",
        }

        return [params1, params2, params3]
