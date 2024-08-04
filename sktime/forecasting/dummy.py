# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dummy forecasters."""

__author__ = ["fkiraly", "geetu040"]

import numpy as np
import pandas as pd

from sktime.datatypes import convert_to
from sktime.forecasting.base import BaseForecaster, _BaseGlobalForecaster


class DummyGlobalForecaster(_BaseGlobalForecaster):
    """Dummy Global Forecaster for time series forecasting.

    This forecaster provides a simple implementation for global forecasting by
    aggregating historical data using a specified aggregation method and repeating
    the aggregated values for future predictions. It is designed to serve as a
    baseline model, allowing for quick experimentation and testing of forecasting
    pipelines without incurring significant computational costs or dependencies
    as this implementation relies solely on sktime's built-in dependencies.

    Parameters
    ----------
    aggregate_method : callable, default=numpy.mean
        A method for aggregating predictions over the autoregressive
        iterations. If not specified, defaults to `numpy.mean`.
    broadcasting : bool, default=True
        multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from `predict`.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.dummy import DummyGlobalForecaster
    >>>
    >>> y = load_airline()
    >>>
    >>> forecaster = DummyGlobalForecaster()
    >>> forecaster.fit(y=y) # doctest: +SKIP
    >>> preds = forecaster.predict(fh=[1, 2, 3]) # doctest: +SKIP
    >>> print(preds) # doctest: +SKIP
    Period
    1961-01    280.298611
    1961-02    280.298611
    1961-03    280.298611
    Freq: M, Name: Number of airline passengers, dtype: float64

    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
    >>> from sktime.forecasting.dummy import DummyGlobalForecaster
    >>>
    >>> # generate and prepare random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> x = data[["c0", "c1"]]
    >>> y = data["c2"].to_frame()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     x, y, test_size=0.2, train_size=0.8, shuffle=False
    ... )
    >>> len_levels = len(y_test.index.names)
    >>> y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
    ...     lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-20]
    ... )
    >>>
    >>> # create forecaster and wrapper
    >>> forecaster = DummyGlobalForecaster()
    >>> forecaster.fit(y=y_train, X=X_train) # doctest: +SKIP
    >>> preds = forecaster.predict(y=y_test, X=X_test, fh=[1, 2, 3]) # doctest: +SKIP
    >>> print(preds) # doctest: +SKIP
                                c2
    h0   h1    time
    h0_4 h1_0  2000-01-31  5.109736
            2000-02-01  5.109736
            2000-02-02  5.109736
        h1_1  2000-01-31  5.383069
            2000-02-01  5.383069
    ...                         ...
        h1_98 2000-02-01  5.053547
            2000-02-02  5.053547
        h1_99 2000-01-31  5.105706
            2000-02-01  5.105706
            2000-02-02  5.105706

    [600 rows x 1 columns]
    >>>
    """

    _tags = {
        "scitype:y": "both",
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "capability:global_forecasting": True,
    }

    def __init__(self, aggregate_method=None, broadcasting=False):
        self.broadcasting = broadcasting
        self.aggregate_method = (
            aggregate_method if aggregate_method is not None else np.mean
        )
        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        pass

    def _predict(self, fh, X=None, y=None):
        _y = self._y if y is None else y

        # convert to multi-index if not
        converted_to_multiindex = False
        if not isinstance(_y.index, pd.MultiIndex):
            _y = pd.DataFrame(
                _y.values,
                index=pd.MultiIndex.from_product(
                    [["h0"], _y.index], names=["h0", _y.index.name]
                ),
                columns=_y.columns,
            )
            converted_to_multiindex = True

        # apply aggregate on the inner-most (timestamps) level
        groupby_levels = list(range(len(_y.index.names) - 1))
        data = _y.groupby(level=groupby_levels).aggregate(self.aggregate_method)

        # inner-most (timestamps) level index
        datetime_index = fh.to_absolute(self._cutoff).to_pandas()

        # expand on the aggregated values
        preds = pd.concat([data] * len(datetime_index), keys=datetime_index)

        # move inner-most (timestamps) level to the end
        new_order = list(range(1, preds.index.nlevels)) + [0]
        preds = preds.reorder_levels(new_order)

        # removed in fh.to_absolute
        preds.index.names = _y.index.names

        # inner-most (timestamps) level indexes are not sorted
        preds = preds.sort_index()

        # convert back from multi-index if needed
        if converted_to_multiindex:
            preds = preds.droplevel(0)

        return preds

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {
                "aggregate_method": np.median,
            },
            {"aggregate_method": np.mean, "broadcasting": True},
        ]


class ForecastKnownValues(BaseForecaster):
    """Forecaster that plays back known or prescribed values as forecasts.

    Takes a data set of "known future values" to produces these in the sktime interface.

    Common use cases for this forecaster:

    * as a dummy or naive forecaster with a known baseline expectation
    * as a forecaster with (non-naive) expert forecasts, "known" values as per expert
    * as a counterfactual in benchmarking experiments, "what if we knew the truth"
    * to pass forecast data values in a composite used for postprocessing,
      e.g., in combination with ReconcilerForecaster for an isolated reconciliation step

    When forecasting, uses ``pandas.DataFrame.reindex`` under the hood to obtain
    predicted
    values from ``y_known``. Parameters other than ``y_known`` are directly passed
    on to ``pandas.DataFrame.reindex``.

    Parameters
    ----------
    y_known : pd.DataFrame or pd.Series in one of the sktime compatible data formats
        should contain known values that the forecaster will replay in predict
        can also be in a non-pandas sktime data format, will then be coerced to pandas
    method : str or None, optional, default=None
        one of {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
        method to use for imputing indices at which forecasts are unavailable in y_known
    fill_value : scalar, optional, default=np.NaN
        value to use for any missing values (e.g., if ``method`` is None)
    limit : int, optional, default=None=infinite
        maximum number of consecutive elements to bfill/ffill if
        ``method=bfill``/``ffill``

    Examples
    --------
    >>> y_known = pd.DataFrame(range(100))
    >>> y_train = y_known[:24]
    >>>
    >>> from sktime.forecasting.dummy import ForecastKnownValues
    >>>
    >>> fcst = ForecastKnownValues(y_known)
    >>> fcst.fit(y_train, fh=[1, 2, 3])
    ForecastKnownValues(...)

    The forecast "plays back" the known/prescribed values from y_known

    >>> y_pred = fcst.predict()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, y_known, method=None, fill_value=None, limit=None):
        self.y_known = y_known
        self.method = method
        self.fill_value = fill_value
        self.limit = limit

        super().__init__()

        PANDAS_DF_TYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]

        self._y_known = convert_to(y_known, PANDAS_DF_TYPES)

        idx = self._y_known.index
        if isinstance(idx, pd.MultiIndex):
            if idx.nlevels >= 3:
                mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
            elif idx.nlevels == 2:
                mtypes = ["pd.DataFrame", "pd-multiindex"]
            self.set_tags(**{"y_inner_mtype": mtypes})
            self.set_tags(**{"X_inner_mtype": mtypes})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # no fitting, we already know the forecast values
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : Point predictions
        """
        reindex_params = {"method": self.method, "limit": self.limit}
        if self.fill_value is not None:
            reindex_params["fill_value"] = self.fill_value

        fh_abs = fh.to_absolute_index(self.cutoff)

        try:
            y_pred = self._y_known.reindex(fh_abs, **reindex_params)
            y_pred = y_pred.reindex(self._y.columns, axis=1, **reindex_params)
        # TypeError happens if indices are incompatible types
        except TypeError:
            if self.fill_value is None:
                y_pred = pd.DataFrame(index=fh_abs, columns=self._y.columns)
            else:
                y_pred = pd.DataFrame(
                    self.fill_value, index=fh_abs, columns=self._y.columns
                )

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.utils._testing.series import _make_series

        y = _make_series(n_columns=1)
        y2 = _make_series(n_columns=2, n_timepoints=15, index_type="int")

        params1 = {"y_known": y, "fill_value": 42}
        params2 = {"y_known": y2, "method": "ffill", "limit": 3, "fill_value": 42}

        return [params1, params2]
