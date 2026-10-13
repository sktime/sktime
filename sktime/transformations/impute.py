#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Transformer to impute missing values in series."""

__author__ = ["aiwalter"]
__all__ = ["Imputer"]

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.datatypes import VectorizedDF
from sktime.transformations.base import BaseTransformer

# Methods whose panel application is broadcast by ``BaseTransformer`` rather than
# implemented on a multi-index frame inside ``Imputer``.
_VECTORIZED_METHODS = ("drift", "linear", "nearest", "random", "forecaster")


class Imputer(BaseTransformer):
    """Missing value imputation.

    The Imputer transforms input series by replacing missing values according
    to an imputation strategy specified by ``method``.

    Parameters
    ----------
    method : str, default="drift"
        Method to fill the missing values. Not all methods can extrapolate, so after
        ``method`` is applied the remaining missing values are filled with ``ffill``
        then ``bfill``.

        * "drift" : drift/trend values by sktime.PolynomialTrendForecaster(degree=1).
          The forecaster is fitted on data seen in ``fit`` (after ``ffill`` then
          ``bfill``) and predicts indices which are missing in ``transform``.
        * "linear" : linear interpolation, uses pd.Series.interpolate()
            WARNING: This method can not extrapolate, so it is fitted always on the
            data given to transform().
        * "nearest" : use nearest value, uses pd.Series.interpolate()
        * "constant" : same constant value (given in arg value) for all NaN
        * "mean" : pd.Series.mean() of data seen in ``fit``
          to use data in transform, wrap this estimator in ``FitInTransform``
        * "median" : pd.Series.median() of data seen in ``fit``
          to use data in transform, wrap this estimator in ``FitInTransform``
        * "backfill" to "bfill" : applies ``pd.Series.bfill`` to all data
        * "pad" or "ffill" : applies ``pd.Series.ffill`` to all data
        * "random" : random values between pd.Series.min() and .max() of the series
          passed to ``transform``, per instance.
          if pd.Series dtype is int, sample is uniform discrete
          if pd.Series dtype is float, sample is uniform continuous
        * "forecaster" : use an sktime forecaster, given in param ``forecaster``.
          First, X seen in ``fit`` is filled with ``ffill`` then ``bfill``
          then forecaster is fitted to filled X, and ``predict`` values are queried
          at indices of X data in ``transform`` which had missing values.
          ``forecaster`` is always applied by variable and instance.

        The following methods fit non-trivially on data seen in ``fit``:
        ``drift``, ``mean``, ``median``, and ``forecaster``.
        ``linear``, ``nearest``, ``constant``, ``ffill``, ``bfill``, ``pad``,
        ``backfill``, and ``random`` do not use values seen in ``fit``.

        ``transform`` accepts a different number of instances than ``fit``.
        Parameters estimated per instance are reused when the instance index
        matches an index seen in ``fit``. For any other instance:

        * ``mean`` and ``median`` fill with the statistic pooled over all
          values seen in ``fit``
        * ``drift`` and ``forecaster`` fit that model on the series passed to
          ``transform``
        * the remaining methods are applied directly to each series in
          ``transform``

    missing_values : str, int, float, regex, list, or None, default=None
        Value to consider as `np.nan`` and impute, passed to ``DataFrame.replace``
        If str, int, float, all entries equal to ``missing_values`` will be imputed,
        in addition to ``np.nan.``
        If regex, all entries matching regex will be imputed, in addition to ``np.nan.``
        If list, must be list of str, int, float, or regex.
        Values matching any list element by above rules will be imputed,
        in addition to ``np.nan``.
        If None, then only ``np.nan`` values are imputed.

    value : int/float, default=None
        Value to use to fill missing values when method="constant".
        Only used if ``method="constant"``, otherwise ignored.

    forecaster : Any Forecaster based on sktime.BaseForecaster, default=None
        Use a given Forecaster to impute by insample predictions when
        ``method="forecaster"``. Before fitting, missing data is imputed with
        ``method="ffill"`` or ``"bfill"`` as heuristic. In case of multivariate X,
        a clone of ``forecaster`` is applied per column.
        Only used if ``method="forecaster"``, otherwise ignored.

    random_state : int/float/str, optional
        Value to set random.seed() if method="random", default None

    Examples
    --------
    >>> from sktime.transformations.impute import Imputer
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> transformer = Imputer(method="drift")
    >>> transformer.fit(y_train)
    Imputer(...)
    >>> import numpy as np
    >>> y_test.iloc[3] = np.nan
    >>> y_hat = transformer.transform(y_test)
    """

    _tags = {
        "authors": ["aiwalter"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
        "capability:missing_values": True,
        "skip-inverse-transform": True,
        "capability:inverse_transform": True,
        "capability:multivariate": True,
        "capability:missing_values:removes": True,
        # is transform result always guaranteed to contain no missing values?
        "remember_data": False,  # remember all data seen as _X
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
        "tests:skip_by_name": ["test_categorical_X_passes"],
    }

    def __init__(
        self,
        method="drift",
        random_state=None,
        value=None,
        forecaster=None,
        missing_values=None,
    ):
        self.method = method
        self.missing_values = missing_values
        self.value = value
        self.forecaster = forecaster
        self.random_state = random_state
        super().__init__()

        # these methods require self._X remembered in _fit and _update
        if method in ["drift", "forecaster", "random"]:
            self.set_tags(**{"remember_data": True})

        # these methods can be applied to multi-index frames without vectorization or
        # by using an efficient pandas native method
        if method in [
            "constant",
            "mean",
            "median",
            "backfill",
            "bfill",
            "pad",
            "ffill",
        ]:
            self.set_tags(
                **{
                    "X_inner_mtype": [
                        "pd.DataFrame",
                        "pd-multiindex",
                        "pd_multiindex_hier",
                    ]
                }
            )

        if method == "forecaster":
            self.set_tags(**{"y_inner_mtype": ["pd.DataFrame"]})

    def fit(self, X, y=None):
        """Fit transformer to X, optionally to y.

        Parameters
        ----------
        X : time series in sktime compatible data container format
            Data to fit transform to.
        y : time series in sktime compatible data container format, optional
            Additional data, e.g., labels for transformation.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # Validate even when a later vectorized ``fit`` only clones this object.
        self._check_method()
        super().fit(X, y)
        # A single series does not create ``transformers_``. Leave vectorization
        # undecided so ``transform`` can broadcast that fit across a panel.
        if (
            self.method in _VECTORIZED_METHODS
            and getattr(self, "_is_vectorized", False) is False
        ):
            self._is_vectorized = "unknown"
        return self

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        self._check_method()
        # all methods of Imputer that are actually doing a fit are
        # implemented here. Some methods don't need to fit, so they are just
        # implemented in _transform

        index = X.index
        if isinstance(index, pd.MultiIndex):
            X_grouped = X.groupby(level=list(range(index.nlevels - 1)))
            if self.method == "mean":
                self._mean = X_grouped.mean()
                # Pooled statistic for instances that were not in ``fit``.
                self._mean_fallback = X.mean()
            elif self.method == "median":
                self._median = X_grouped.median()
                self._median_fallback = X.median()
        else:
            if self.method in ["drift", "forecaster"]:
                self._y = y.copy() if y is not None else None
                if self.method == "drift":
                    from sktime.forecasting.trend import PolynomialTrendForecaster

                    self._forecaster = PolynomialTrendForecaster(degree=1)
                elif self.method == "forecaster":
                    self._forecaster = self.forecaster.clone()
            elif self.method == "mean":
                self._mean = X.mean()
            elif self.method == "median":
                self._median = X.median()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        X : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        X = X.copy()

        # replace missing_values with np.nan
        if self.missing_values:
            X = X.replace(to_replace=self.missing_values, value=np.nan)

        if not _has_missing_values(X):
            return X

        index = X.index

        if self.method == "random":
            for col in X.columns:
                isna = X[col].isna()
                X.loc[isna, col] = self._create_random_distribution(X[col])(isna.sum())
            return X
        elif self.method == "constant":
            return X.fillna(value=self.value)
        elif isinstance(index, pd.MultiIndex):
            X_group_levels = list(range(index.nlevels - 1))

            if self.method in ["backfill", "bfill"]:
                X = X.groupby(level=X_group_levels).bfill()
            elif self.method in ["pad", "ffill"]:
                X = X.groupby(level=X_group_levels).ffill()
            elif self.method == "mean":
                X = self._fillna_from_fit_statistic(
                    X, self._mean, getattr(self, "_mean_fallback", None)
                )
            elif self.method == "median":
                X = self._fillna_from_fit_statistic(
                    X, self._median, getattr(self, "_median_fallback", None)
                )
            else:
                raise AssertionError("Code should not be reached")

            # fill first/last elements of series,
            # as some methods can't impute those
            X = X.groupby(level=X_group_levels).ffill()
            X = X.groupby(level=X_group_levels).bfill()

            return X

        else:
            if self.method in ["backfill", "bfill"]:
                X = X.bfill()
            elif self.method in ["pad", "ffill"]:
                X = X.ffill()
            elif self.method == "drift":
                X = self._impute_with_forecaster(X, y)
            elif self.method == "forecaster":
                X = self._impute_with_forecaster(X, y)
            elif self.method == "mean":
                stat = self._statistic_for_unindexed(self._mean, "mean")
                return X.fillna(value=self._align_statistic(stat, X.columns))
            elif self.method == "median":
                stat = self._statistic_for_unindexed(self._median, "median")
                return X.fillna(value=self._align_statistic(stat, X.columns))
            elif self.method in ["nearest", "linear"]:
                X = X.interpolate(method=self.method)
            else:
                raise ValueError(f"`method`: {self.method} not available.")

            # fill first/last elements of series,
            # as some methods (e.g. "linear") can't impute those
            X = X.ffill().bfill()

            return X

    def _vectorize(self, methodname, **kwargs):
        """Vectorize, allowing panels whose instance set differs from ``fit``."""
        X = kwargs.get("X")
        if methodname == "transform" and self._instance_map_does_not_apply(X):
            return self._transform_unaligned_instances(X, kwargs.get("y"))
        return super()._vectorize(methodname, **kwargs)

    def _instance_map_does_not_apply(self, X):
        """Return True when fitted per-instance models cannot be zipped onto ``X``.

        Equal-sized panels keep the base class positional broadcast. A single
        series, or a panel with a different number of instances, does not.
        """
        if self.get_tag("fit_is_empty"):
            return False
        fitted = getattr(self, "transformers_", None)
        if isinstance(X, pd.DataFrame):
            # ``len`` of a series frame is the number of time points, not instances.
            return fitted is not None
        if fitted is None:
            return isinstance(X, VectorizedDF)
        if not isinstance(X, VectorizedDF):
            return False
        return len(X) != int(np.prod(fitted.shape))

    def _lookup_fitted_transformer(self, fitted, key):
        """Return the fitted per-instance imputer for ``key``, or None."""
        if fitted is None or key not in fitted.index:
            return None
        cell = fitted.loc[key]
        if isinstance(cell, pd.DataFrame):
            cell = cell.iloc[0]
        return cell.iloc[0]

    def _transform_unaligned_instances(self, X, y=None):
        """Impute when transform instances are not the fitted instance map.

        Instances whose index was seen in ``fit`` reuse that fitted imputer.
        Every other series is imputed on its own, which is valid for methods
        that do not need the identity of a training instance.
        """
        if isinstance(X, pd.DataFrame):
            fresh = self.clone()
            return fresh.fit(X, y).transform(X, y)

        y_items = list(y.items()) if isinstance(y, VectorizedDF) else None
        fitted = getattr(self, "transformers_", None)
        Xts = []
        for i, (key, _, inst) in enumerate(X.items()):
            y_inst = None if y_items is None else y_items[i][2]
            est = self._lookup_fitted_transformer(fitted, key)
            if est is not None:
                Xts.append(est.transform(inst, y_inst))
            elif fitted is None:
                # One series was fitted; broadcast that model to each instance.
                Xts.append(self._transform_broadcast_series(inst, y_inst))
            else:
                fresh = self.clone()
                Xts.append(fresh.fit(inst, y_inst).transform(inst, y_inst))
        return X.reconstruct(Xts, overwrite_index=False)

    def _transform_broadcast_series(self, inst, y_inst):
        """Apply a model fitted on one series to a single panel instance.

        Numpy series use a positional column name such as ``0``, while numpy
        panels use ``var_0``. When the width matches, align names for the call.
        """
        renamed = None
        y_renamed = None
        try:
            X_fit = getattr(self, "_X", None)
            if (
                isinstance(X_fit, pd.DataFrame)
                and len(X_fit.columns) == len(inst.columns)
                and not X_fit.columns.equals(inst.columns)
            ):
                renamed = X_fit.columns.copy()
                X_fit.columns = inst.columns
                y_fit = getattr(self, "_y", None)
                if (
                    isinstance(y_fit, pd.DataFrame)
                    and len(y_fit.columns) == len(inst.columns)
                ):
                    y_renamed = y_fit.columns.copy()
                    y_fit.columns = inst.columns
            return self._transform(inst, y_inst)
        finally:
            if renamed is not None:
                self._X.columns = renamed
            if y_renamed is not None:
                self._y.columns = y_renamed

    def _align_statistic(self, stat, columns):
        """Reindex a column-wise statistic when only the column names differ."""
        if (
            isinstance(stat, pd.Series)
            and len(stat) == len(columns)
            and not stat.index.equals(columns)
        ):
            return pd.Series(stat.to_numpy(), index=columns)
        return stat

    def _fillna_from_fit_statistic(self, X, stat, fallback):
        """Fill missing values from per-instance ``stat``, else pooled ``fallback``.

        Instances present in ``stat`` never take ``fallback``. An instance that
        was entirely missing in ``fit`` therefore stays missing.
        """
        if not isinstance(stat, pd.DataFrame):
            stat = self._align_statistic(stat, X.columns)
            return X.fillna(value=stat)

        seen = np.asanyarray(X.index.droplevel(-1).isin(stat.index))
        if seen.all():
            return X.fillna(value=stat)

        filled = X.fillna(value=stat)
        if (~seen).any():
            fallback = self._align_statistic(fallback, X.columns)
            filled.loc[~seen] = X.loc[~seen].fillna(value=fallback)
        return filled

    def _statistic_for_unindexed(self, stat, kind):
        """Return a column-wise statistic that can fill a single series.

        Per-instance statistics are indexed by instance and do not align with a
        plain time index. Use the pooled statistic in that case.
        """
        if not isinstance(stat, pd.DataFrame):
            return stat
        fallback_name = f"_{kind}_fallback"
        if hasattr(self, fallback_name):
            return getattr(self, fallback_name)
        return stat

    def _check_method(self):
        method = self.method
        if method not in [
            "mean",
            "drift",
            "linear",
            "nearest",
            "constant",
            "median",
            "backfill",
            "bfill",
            "pad",
            "ffill",
            "random",
            "forecaster",
        ]:
            raise ValueError(f"Given method {method} is not an allowed method.")
        if method == "constant" and self.value is None:
            raise ValueError(
                """Imputing with method=\"constant\" can only be used if parameter
                value" is not None"""
            )
        elif method == "forecaster" and self.forecaster is None:
            raise ValueError(
                """Imputing with method=\"forecaster\" can only be
                used if param forecaster is not None"""
            )
        else:
            pass

    def _create_random_distribution(self, z: pd.Series):
        """Create uniform distribution function within boundaries of given series.

        The distribution is discrete, if the series contains only int-like values.

        Parameters
        ----------
        z : pd.Series
            A series to create a random distribution from

        Returns
        -------
        Callable[[Optional[int]], float]
            Random (discrete) uniform distribution between min and max of series
        """
        rng = check_random_state(self.random_state)
        if (z.dropna() % 1 == 0).all():
            return lambda size, low=z.min(), high=z.max(): rng.randint(
                low=low, high=high, size=size
            )
        else:
            return lambda size, low=z.min(), high=z.max(): rng.uniform(
                low=low, high=high, size=size
            )

    def _impute_with_forecaster(self, X, y):
        """Use a given forecaster for imputation by in-sample predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Series to impute.
        y : pd.DataFrame
            Exog data for forecaster.

        Returns
        -------
        Xt : pd.DataFrame
            Series with imputed values.
        """
        from sktime.forecasting.base import ForecastingHorizon

        for col in X.columns:
            if _has_missing_values(X[col]):
                # define fh based on index of missing values
                na_index = X[col].index[X[col].isna()]
                fh = ForecastingHorizon(values=na_index, is_relative=False)

                # fill NaN before fitting with ffill and backfill (heuristic)

                self._forecaster.fit(
                    y=self._X[col].ffill().bfill(),
                    X=self._y[col].ffill().bfill() if self._y is not None else None,
                    fh=fh,
                )

                # replace missing values with predicted values
                X.loc[na_index, col] = self._forecaster.predict(fh=fh, X=y)
        return X

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
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import TrendForecaster

        linear_forecaster = make_reduction(LinearRegression(), strategy="multioutput")

        return [
            {"method": "drift"},
            {"method": "linear"},
            {"method": "nearest"},
            {"method": "constant", "value": 1},
            {"method": "median"},
            {"method": "backfill"},
            {"method": "bfill"},
            {"method": "pad"},
            {"method": "random"},
            {"method": "forecaster", "forecaster": TrendForecaster()},
            {"method": "forecaster", "forecaster": linear_forecaster},
        ]


def _has_missing_values(X):
    return X.isnull().to_numpy().any()
