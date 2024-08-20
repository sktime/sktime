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
        An instance of a forecasting model that is compatible with
        the `AutoRegressiveWrapper`. This forecaster must support
        the methods required for fitting and predicting.
    # TODO: None
    horizon_length : int
        The length of the forecasting horizon, defining how many time
        steps into the future the forecaster should predict with each
        call to the `predict` method.
    aggregate_method : callable, default=numpy.mean
        A method for aggregating predictions over the autoregressive
        iterations. If not specified, defaults to `numpy.mean`.

    Example
    -------
    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    >>> from sktime.forecasting.compose import AutoRegressiveWrapper
    >>>
    >>> y = load_airline()
    >>> forecaster = PytorchForecastingTFT(trainer_params={
    ...     "max_epochs": 5,
    ... }) # doctest: +SKIP
    >>> wrapper = AutoRegressiveWrapper(
    ...     forecaster=forecaster,
    ...     horizon_length=3, # prediction_length for forecaster
    ...     aggregate_method=np.mean,
    ... ) # doctest: +SKIP
    >>>
    >>> wrapper.fit(y) # doctest: +SKIP
    >>>
    >>> # forecast multiple forecasting horizons with same trained model
    >>> preds = wrapper.predict(fh=[1, 5]) # doctest: +SKIP
    >>> preds = wrapper.predict(fh=[8, 9, 10, 11, 12]) # doctest: +SKIP

    >>> import numpy as np
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    >>> from sktime.forecasting.compose import AutoRegressiveWrapper
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
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
    >>> forecaster = PytorchForecastingTFT(trainer_params={
    ...     "max_epochs": 1,
    ... }) # doctest: +SKIP
    >>> wrapper = AutoRegressiveWrapper(
    ...     forecaster=forecaster,
    ...     horizon_length=3, # prediction_length for forecaster
    ...     aggregate_method=np.mean,
    ... ) # doctest: +SKIP
    >>> wrapper.fit(y) # doctest: +SKIP
    >>>
    >>> # forecast multiple forecasting horizons with same trained model
    >>>
    >>> wrapper.predict(fh=[1, 5]) # doctest: +SKIP
                                c2
    h0   h1    time
    h0_0 h1_0  2000-02-20  5.278717
            2000-02-24  5.303032
        h1_1  2000-02-20  5.106282
            2000-02-24  5.119611
        h1_10 2000-02-20  4.924220
    ...                         ...
    h0_4 h1_97 2000-02-24  4.815347
        h1_98 2000-02-20  5.022863
            2000-02-24  5.044283
        h1_99 2000-02-20  4.714049
            2000-02-24  4.768126

    [2000 rows x 1 columns]
    >>>
    >>> wrapper.predict(fh=[8, 9, 10, 11, 12]) # doctest: +SKIP
                                c2
    h0   h1    time
    h0_0 h1_0  2000-02-27  5.312222
            2000-02-28  5.323535
            2000-02-29  5.351599
            2000-03-01  5.367970
            2000-03-02  5.379033
    ...                         ...
    h0_4 h1_99 2000-02-27  4.754828
            2000-02-28  4.766777
            2000-02-29  4.872816
            2000-03-01  4.886779
            2000-03-02  4.896248

    [5000 rows x 1 columns]
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
            # TODO: improve the error message
            raise ValueError("forecaster should be global")

    def _fit(self, y, X=None, fh=None):
        if self.horizon_length is None and fh is None:
            # TODO: improve message here
            raise ValueError("Both horizon_length and fh cannot be None")

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
