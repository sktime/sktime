# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""AutoRegressiveWrapper for Forecasters with fixed prediction length."""

__author__ = ["geetu040"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.utils.dependencies import _check_soft_dependencies


class AutoRegressiveWrapper(_BaseGlobalForecaster):
    """
    AutoRegressiveWrapper for Forecasters with Fixed Prediction Length.

    This class acts as a wrapper for existing forecasting models, enabling them
    to perform autoregressive predictions when they are originally designed
    for fixed-length outputs. It addresses the limitations of certain
    forecasters that can only predict a fixed number of future time steps,
    such as the KanForecaster and many reduction-based forecasters.

    By utilizing an autoregressive approach, this wrapper allows the forecaster
    to make predictions iteratively. Each prediction is fed back into the
    model as part of the input for subsequent predictions, which can significantly
    enhance flexibility and efficiency, especially when training time is a
    consideration and when dealing with varying forecasting horizons.

    The `AutoRegressiveWrapper` is particularly useful in scenarios where:
    - The underlying forecaster takes considerable time to train with
      increased output sizes.
    - The forecaster is integrated with probabilistic wrappers (e.g.,
      Conformal Intervals) that require variable forecasting horizons.

    The wrapped forecaster must inherit from `_BaseGlobalForecaster`, ensuring
    that it supports the necessary interface for fitting and predicting.

    Attributes
    ----------
    forecaster : _BaseGlobalForecaster
        An instance of a global forecasting model.
        Wrapping a forecaster with this class is beneficial when:
        - Variable Forecasting Horizons: The forecasting task requires
          predictions over a varying number of future time steps, which
          the underlying forecaster cannot handle natively.
        - Training Efficiency: The forecaster is computationally expensive
          to train, especially when the output size increases. By using
          the autoregressive approach, the model can be trained more efficiently
          on smaller output sizes while still producing long-horizon forecasts.
    horizon_length : int, default=None
        Specifies the number of time steps into the future that the forecaster
        should predict in each call to the `predict` method.
        - If set to an integer value, this determines the fixed length of the
          forecasting horizon.
        - If set to None (the default), the `horizon_length` is automatically
          determined based on the provided `fh` (forecasting horizon) during the
          fitting process. The maximum value in `fh` will be used as the
          `horizon_length`.
        Note: If `horizon_length` is None and `fh` is not provided during the
        fitting stage, a `ValueError` will be raised. In such cases, you must
        either specify `horizon_length` or provide a valid `fh` when calling `fit`.
    aggregate_method : callable, default=numpy.mean
        A method for aggregating predictions over the autoregressive
        iterations. If not specified, defaults to `numpy.mean`.

    Example
    -------
    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats
    >>> from sktime.forecasting.compose import AutoRegressiveWrapper
    >>> from sktime.utils.plotting import plot_series
    >>>
    >>> # prepare data
    >>> y = load_airline()
    >>>
    >>> # create forecaster and wrapper
    >>> forecaster = PytorchForecastingNBeats(trainer_params={
    ...     "max_epochs": 20,
    ... }) # doctest: +SKIP
    >>> wrapper = AutoRegressiveWrapper(
    ...     forecaster=forecaster,
    ...     horizon_length=5, # prediction_length for forecaster
    ...     aggregate_method=np.mean,
    ... ) # doctest: +SKIP
    >>> wrapper.fit(y) # doctest: +SKIP
    >>>
    >>> # forecast multiple forecasting horizons with same trained model
    >>> y_pred = wrapper.predict(fh=[1, 5]) # doctest: +SKIP
    >>> y_pred = wrapper.predict(fh=[8, 9, 10, 11, 12]) # doctest: +SKIP
    >>> y_pred = wrapper.predict(fh=list(range(1, 30))) # doctest: +SKIP
    >>>
    >>> # plot the forecasts
    >>> plot_series(y, y_pred) # doctest: +SKIP

    >>> import numpy as np
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR
    >>> from sktime.forecasting.compose import AutoRegressiveWrapper
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
    >>> from sktime.utils.plotting import plot_series
    >>>
    >>> # generate and prepare random data
    >>> max_pred_len = 20
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 100),
    ...     max_timepoints=100,
    ...     min_timepoints=100,
    ...     n_columns=3
    ... )
    >>> X = data[["c0", "c1"]]
    >>> y = data["c2"].to_frame()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, shuffle=False
    ... )
    >>> y_true = y_test.groupby(level=[0,1]).apply(
    ...     lambda x: x.droplevel([0,1]).iloc[-max_pred_len:]
    ... )
    >>> y_test = y_test.groupby(level=[0,1]).apply(
    ...     lambda x: x.droplevel([0,1]).iloc[:-max_pred_len]
    ... )
    >>>
    >>> # create forecaster and wrapper
    >>> forecaster = PytorchForecastingDeepAR(
    ...     trainer_params={"max_epochs": 1},
    ...     broadcasting=False,
    ... ) # doctest: +SKIP
    >>> wrapper = AutoRegressiveWrapper(
    ...     forecaster=forecaster,
    ...     horizon_length=3, # prediction_length for forecaster
    ...     aggregate_method=np.mean,
    ... ) # doctest: +SKIP
    >>> wrapper.fit(y=y_train, X=X_train) # doctest: +SKIP
    >>>
    >>> # forecast multiple forecasting horizons with same trained model
    >>> y_pred = wrapper.predict(fh=[1, 5]) # doctest: +SKIP
    >>> y_pred = wrapper.predict(y=y_test, X=X_test, fh=[1, 2, 4, 6]) # doctest: +SKIP
    >>> y_pred = wrapper.predict(
    ...     y=y_test,
    ...     X=X_test,
    ...     fh=[8, 9, 10, 11, 12]
    ... ) # doctest: +SKIP
    >>> y_pred = wrapper.predict(
    ...     y=y_test,
    ...     X=X_test,
    ...     fh=list(range(1, max_pred_len))
    ... ) # doctest: +SKIP
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
        "capability:insample": False,
        "capability:global_forecasting": True,
    }

    def __init__(self, forecaster, horizon_length=None, aggregate_method=None):
        super().__init__()

        self.forecaster = forecaster
        self.horizon_length = horizon_length
        self._horizon_length = None
        self.aggregate_method = (
            aggregate_method if aggregate_method is not None else np.mean
        )

        gt = forecaster.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gt:
            raise ValueError(
                f"Error in {self.__class__.__name__}: "
                f"The provided forecaster, {self.forecaster.__class__.__name__}, "
                "does not have global forecasting capabilities. "
                "To use the AutoRegressiveWrapper, "
                "the forecaster must be global in nature. "
                "Ensure that the forecaster has the "
                "'capability:global_forecasting' tag set to True. "
                "You can verify this by calling "
                "forecaster.get_tag('capability:global_forecasting')."
            )

    def _fit(self, y, X=None, fh=None):
        if self.horizon_length is None and fh is None:
            raise ValueError(
                f"Error in {self.__class__.__name__}, "
                "Either 'horizon_length' must be specified during initialization or "
                "'fh' must be provided during fit. Both cannot be None."
            )
        if fh is not None:
            max_fh = max(fh.to_relative(self._cutoff)._values)
            if self.horizon_length is None:
                self._horizon_length = max_fh
            else:
                self._horizon_length = max(max_fh, self.horizon_length)
        else:
            self._horizon_length = self.horizon_length

        # create a new fh for forecaster
        _fh = ForecastingHorizon(range(1, self._horizon_length + 1))

        self.forecaster.fit(y=y, X=X, fh=_fh)

    def _predict(self, fh, X=None, y=None):
        # use fh to find the maximum length to forecast regressively
        max_fh = max(
            *fh.to_relative(self._cutoff)._values,
            self._horizon_length,
        )

        hist_y = self._y if y is None else y
        hist_y = deepcopy(hist_y)  # make copy because we are going to add values to it

        # calculate initial length of hist_y
        if not isinstance(hist_y.index, pd.MultiIndex):
            initial_length = len(hist_y.index)
        else:
            initial_length = len(hist_y.index.levels[-1])

        if y is None and X is not None:
            # special case of non global forecasting
            # here X contains only (future) values
            # however for the interface, X should contain (hist + future) values
            # since this is non-global forecasting, we can use hist from self._X
            X = pd.concat([self._X, X])

        for i in range(max_fh - self._horizon_length + 1):
            # truncate to the needed lengths
            _y = self._truncate(hist_y, end_index=initial_length + i)
            _x = self._truncate(X, end_index=initial_length + i + self._horizon_length)

            # forecast for self._horizon_length
            preds = self.forecaster.predict(y=_y, X=_x)

            # use predicted values as history for next iteration
            hist_y = self._concate_preds(hist_y, preds)

        # collect forecasting points from hist_y
        # code works for all indexes
        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = hist_y.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        preds = hist_y.loc[dateindex]

        return preds

    def _truncate(self, data, end_index):
        # keep rows till this index in data

        if data is None:
            # when there is no X (exogenous data)
            return None

        if not isinstance(data.index, pd.MultiIndex):
            return data.head(end_index)

        # truncating for Multi-Index
        # levels other than the timestamps (innermost/base level)
        groupby_levels = list(range(len(data.index.names) - 1))
        data = data.groupby(
            level=groupby_levels,
            group_keys=True,  # to keep stable conversion with pandas<2.0.0
        ).apply(lambda x: x.head(end_index))
        data.index = data.index.droplevel(groupby_levels)
        return data

    def _concate_preds(self, _y, preds):
        _y = pd.concat([_y, preds])
        _y = _y.groupby(level=list(range(_y.index.nlevels)))
        _y = _y.aggregate(self.aggregate_method)
        return _y

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
        test_params = []

        # test on DummyGlobalForecaster
        from sktime.forecasting.dummy import DummyGlobalForecaster

        forecaster_test_params = DummyGlobalForecaster.get_test_params()
        for forecaster_test_param in forecaster_test_params:
            test_params.append(
                {
                    "forecaster": DummyGlobalForecaster(**forecaster_test_param),
                    "horizon_length": 3,
                    "aggregate_method": np.mean,
                }
            )

        # test on PytorchForecastingNBeats
        if _check_soft_dependencies("pytorch-forecasting", severity="none"):
            from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats

            forecaster_test_param = PytorchForecastingNBeats.get_test_params()[0]
            forecaster_test_param.update(
                {
                    "broadcasting": False,
                }
            )
            test_params.append(
                {
                    "forecaster": PytorchForecastingNBeats(**forecaster_test_param),
                    "horizon_length": 3,
                    "aggregate_method": np.mean,
                }
            )

        return test_params
