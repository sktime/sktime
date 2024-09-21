#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adaptor for applying Scikit-learn-like transformers to time series."""

__author__ = ["mloning", "fkiraly"]
__all__ = ["TabularToSeriesAdaptor"]

from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.transformations.base import BaseTransformer
from sktime.utils.sklearn import prep_skl_df


class TabularToSeriesAdaptor(BaseTransformer):
    """Adapt scikit-learn transformation interface to time series setting.

    This is useful for applying scikit-learn :term:`tabular` transformations
    to :term:`series <Time series>`, but only works with transformations that
    do not require multiple :term:`instances <instance>` for fitting.

    The adaptor behaves as follows.

    If ``fit_in_transform = False`` and ``X`` is a series
    (``pd.DataFrame``, ``pd.Series``, ``np.ndarray``):

        * ``fit(X)`` fits a clone of ``transformer`` to X (considered as a table)
        * ``transform(X)`` applies transformer.transform to X and returns the result
        * ``inverse_transform(X)`` applies ``transformer.inverse_transform`` to ``X``

    If ``fit_in_transform = True`` and ``X`` is a series
    (``pd.DataFrame``, ``pd.Series``, ``np.ndarray``):

        * ``fit`` is empty
        * ``transform(X)`` applies ``transformer.fit(X).transform(X)`` to ``X``,
          considered as a table, and returns the result
        * ``inverse_transform(X)`` applies ``transformer.fit(X).inverse_transform(X)``
          to ``X``

    If ``fit_in_transform = False``, and ``X`` is of a panel/hierarchical type:
        * ``fit(X)`` fits a clone of ``transformer`` for each individual
          series ``x`` in ``X``
        * ``transform(X)`` applies ``transform(x)`` of the clone belonging to ``x``
          (where the index of x in transform equals the index of x in fit),
          for each individual series ``x`` in ``X``, and returns the result
        * ``inverse_transform(X)`` applies ``transform(x)`` of the clone belonging to
          ``x`` (where the index of x in transform equals the index of ``x`` in fit),
          for each individual series ``x`` in ``X``, and returns the result
        * Note: instances indices in ``transform/inverse_transform``
          must be equal to those seen in ``fit``

    If ``fit_in_transform = True``, and ``X`` is of a panel/hierarchical type:
        * ``fit`` is empty
        * ``transform(X)`` applies ``transformer.fit(x).transform(x)``
          to all individual series ``x`` in ``X`` and returns the result
        * ``inverse_transform(X)`` applies ``transformer.fit(x).inverse_transform(x)``
          to all individual series ``x`` in ``X`` and returns the result

    WARNING: if ``fit_in_transform`` is set to ``False``,
        when applied to Panel or Hierarchical data,
        the resulting transformer will identify individual series in test set
        with series indices in training set, on which instances were fit
        in particular, transform will not work if number of instances
        and indices of instances in transform are different from those in fit

    WARNING: if ``fit_in_transform`` is set to ``True``,
        then each series in the test set will be transformed as batch by fit-predict,
        this may cause information leakage in a forecasting setting
        (but not in a time series classification/regression/clustering setting,
        because in these settings the independent samples are the individual series)

    Whether ``y`` is passed to transformer methods is controlled by ``pass_y``.
    If the inner transformer has non-defaulting ``y`` args, the default behaviour is
    to pass ``y`` to ``fit``, ``fit_transform``, or ``transform``.
    If no ``y`` arg is present, or if it has a default value, ``y`` is not passed.

    If the passed transformer accepts only ``y`` in ``fit`` and ``transform``,
    then ``pass_y`` is ignored, and ``X`` is plugged into the ``y`` argument.

    Parameters
    ----------
    transformer : ``sklearn`` transformer, ``BaseEstimator`` descendant instance
        scikit-learn-like transformer to fit and apply to series.
        This is used as a "blueprint" and not fitted or otherwise mutated.

    fit_in_transform: bool, optional, default=False
        whether transformer_ should be fitted in transform (True), or in fit (False).

        * recommended setting in forecasting (single series or hierarchical): ``False``
        * recommended setting in ts classification, regression, clustering: ``True``

    pass_y : str, optional, one of "auto" (default), "fit", "always", "never"
        Whether to pass y to transformer methods of the ``transformer`` clone.

        * "auto": passes ``y`` to methods ``fit``, ``transform``, ``fit_transform``,
          ``inverse_transform``,
          if and only if ``y`` is a named arg of either method without default.
          Note: passes ``y`` even if it is ``None``
        * "fit": passes ``y`` to method ``fit``, but not to ``transform``.
          Note: passes ``y`` even if it is ``None``, or if not a named arg
        * "always": passes ``y`` to all methods, ``fit``, ``transform``,
          ``inverse_transform``.
          Note: passes ``y`` even if it is ``None``, or if not a named arg
        * "never": never passes ``y`` to any method.

    input_type : str, one of "numpy" (default), "pandas"
        type of data passed to the ``sklearn`` transformer

        * "numpy": 2D ``np.ndarray``
        * "pandas": ``pd.DataFrame``, with column names passed to transformer.
          column names are coerced to strings if not already,
          row index is reset to ``RangeIndex``.

    pooling : str, one of "local" (default), "global"
        whether to apply transformer to each series individually (local),
        or to all series at once (global)

        * "local": applies transformer to each series individually
        * "global": applies transformer to all series at once, pooled
          to a single 2D ``np.ndarray`` or ``pd.DataFrame``

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
        # packaging info
        # --------------
        "authors": ["mloning", "fkiraly"],
        # estimator type
        # --------------
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

    def __init__(
        self,
        transformer,
        fit_in_transform=False,
        pass_y="auto",
        input_type="numpy",
        pooling="local",
    ):
        self.transformer = transformer
        self.transformer_ = clone(self.transformer)
        self.fit_in_transform = fit_in_transform
        self.pass_y = pass_y
        self.input_type = input_type
        self.pooling = pooling

        self._trafo_has_X = self._trafo_has_param_and_default("fit", "X")[0]

        super().__init__()

        if hasattr(transformer, "_get_tags"):
            categorical_list = ["categorical", "1dlabels", "2dlabels"]
            tag_values = transformer._get_tags()["X_types"]
            if any(val in tag_values for val in categorical_list):
                self.set_tags(**{"capability:categorical_in_X": True})

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

        trafo_has_y, trafo_has_y_default = self._trafo_has_param_and_default("fit", "y")
        need_y = trafo_has_y and not trafo_has_y_default
        if need_y or pass_y not in ["auto", "no"]:
            self.set_tags(**{"y_inner_mtype": "numpy1D"})

        if not self._trafo_has_X:
            self.set_tags(**{"y_inner_mtype": "None"})
            self.set_tags(**{"univariate-only": True})

        if pooling == "local":
            self.set_tags(**{"scitype:instancewise": True})
            if input_type == "numpy":
                self.set_tags(
                    **{
                        "X_inner_mtype": "np.ndarray",
                        # categorical is not supported in numpy yet.
                        "capability:categorical_in_X": False,
                    }
                )
            elif input_type == "pandas":
                self.set_tags(**{"X_inner_mtype": "pd.DataFrame"})
            else:
                raise ValueError(
                    "Error in TabularToSeriesAdaptor: "
                    f"input_type={input_type} not supported, must be one of "
                    "'numpy', 'pandas'"
                )
        elif pooling == "global":
            self.set_tags(**{"scitype:instancewise": False})
            PANDAS_TYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
            self.set_tags(**{"X_inner_mtype": PANDAS_TYPES})
        else:
            raise ValueError(
                "Error in TabularToSeriesAdaptor: "
                f"pooling={pooling} not supported, must be one of 'local', 'global'"
            )

    def _trafo_has_param_and_default(self, method="fit", arg="y"):
        """Return if transformer.method has a parameter, and whether it has a default.

        Parameters
        ----------
        method : str, optional, default="fit"
            method name to check
        arg : str, optional, default="y"
            parameter name to check

        Returns
        -------
        has_param : bool
            whether the method ``method`` has a parameter with name ``arg``
        has_default : bool
            whether the parameter ``arg`` of method ``method`` has a default value
        """
        method_fun = getattr(self.transformer, method)
        method_params = list(signature(method_fun).parameters.keys())
        if arg in method_params:
            param = signature(self.transformer.fit).parameters[arg]
            default = param.default
            has_default = default is not param.empty
            return True, has_default
        else:
            return False, False

    def _get_args(self, X, y, method="fit"):
        """Get kwargs for method, depending on pass_y and method.

        The return is a dict which is passed to the method of name method.
        """
        input_type = self.input_type

        if input_type == "numpy" and isinstance(X, pd.DataFrame):
            X = X.values
        if input_type == "pandas" and isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if input_type == "pandas":
            X = X.reset_index(drop=True)
            X = prep_skl_df(X)

        if not self._trafo_has_X:
            return {"y": X}

        pass_y = self.pass_y

        if pass_y == "auto":
            has_y, has_y_default = self._trafo_has_param_and_default(method, "y")
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
            return {"X": X, "y": y}
        else:
            return {"X": X}

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
        fit_args = self._get_args(X, y, method="fit")

        if not self._skip_fit:
            self.transformer_.fit(**fit_args)

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
        # if DataFrame, remember index for later to restore on Xt
        was_df = isinstance(X, pd.DataFrame)
        if was_df:
            saved_index = X.index

        # get args for fit and transform
        fit_args = self._get_args(X, y, method="fit")
        trafo_args = self._get_args(X, y, method="transform")

        # apply transformer
        if self._skip_fit:
            Xt = self.transformer_.fit(**fit_args).transform(**trafo_args)
        else:
            Xt = self.transformer_.transform(**trafo_args)

        # converting to dense if the transformer output was in sparse format
        # Example: sklearn OneHotEncoder's default output is sparse
        if str(type(Xt)) == "<class 'scipy.sparse._csr.csr_matrix'>":
            Xt = Xt.todense()
        # coerce sensibly to 2D np.ndarray
        if isinstance(Xt, (int, float, str)):
            Xt = np.array([[Xt]])
        if not isinstance(Xt, (np.ndarray, pd.DataFrame)):
            Xt = np.array(Xt)
        if Xt.ndim == 1 and hasattr(Xt, "reshape"):
            Xt = Xt.reshape((len(X), 1))

        # restore index if DataFrame
        if was_df:
            if isinstance(Xt, pd.DataFrame):
                Xt.index = saved_index
            else:
                Xt = pd.DataFrame(Xt, index=saved_index)

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
        fit_args = self._get_args(X, y, method="fit")
        it_args = self._get_args(X, y, method="inverse_transform")

        if self.fit_in_transform:
            Xt = self.transformer_.fit(X, **fit_args).inverse_transform(X, **it_args)
        else:
            Xt = self.transformer_.inverse_transform(**it_args)
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        params1 = {"transformer": StandardScaler(), "fit_in_transform": False}
        params2 = {
            "transformer": StandardScaler(),
            "fit_in_transform": True,
            "pass_y": "auto",
        }
        params3 = {"transformer": VarianceThreshold(), "pass_y": "fit"}
        params4 = {"transformer": VarianceThreshold()}
        params5 = {"transformer": LabelEncoder(), "fit_in_transform": True}
        params6 = {
            "transformer": StandardScaler(),
            "pooling": "global",
            "input_type": "pandas",
        }
        params7 = {
            "transformer": StandardScaler(),
            "pooling": "local",
            "input_type": "pandas",
        }
        params8 = {
            "transformer": StandardScaler(),
            "pooling": "global",
            "input_type": "numpy",
        }

        return [params1, params2, params3, params4, params5, params6, params7, params8]


class PandasTransformAdaptor(BaseTransformer):
    """Adapt pandas transformations to sktime interface.

    In ``transform``, executes ``pd.DataFrame`` method of name ``method`` on data,
    optionally with keywords arguments passed, via ``kwargs`` hyper-parameter.
    The ``apply_to`` parameter controls what the data is upon which ``method`` is
    called:
    "call" = for ``X`` seen in ``transform``, "all"/"all_subset" = all data seen so far.
    See below for details.

    For hierarchical series, operation is applied by instance.

    Parameters
    ----------
    method : str, optional, default = None = identity transform
        name of the method of DataFrame that is applied in transform
    kwargs : dict, optional, default = empty dict (no kwargs passed to method)
        arguments passed to DataFrame.method
    apply_to : str, one of "call", "all", "all_subset", optional, default = "call"
        "call" = method is applied to ``X`` seen in transform only
        "all" = method is applied to all ``X`` seen in ``fit``, ``update``,
        ``transform``
            more precisely, the application to ``self._X`` is returned
        "all_subset" = method is applied to all ``X`` like for "all" value,
            but before returning, result is sub-set to indices of ``X`` in ``transform``
        in "all", "all_subset", ``X`` seen in ``transform`` do not update ``self._X``

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
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"method": "diff"}
        params2 = {"method": "diff", "kwargs": {"periods": 2}, "apply_to": "all_subset"}
        params3 = {
            "method": "shift",
            "kwargs": {"periods": 12},
            "apply_to": "all",
        }

        return [params1, params2, params3]
