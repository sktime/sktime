"""
Unified high-level interface for various time series related learning tasks.
"""

from sklearn.base import _pprint
from sklearn.utils.validation import check_is_fitted
from inspect import signature
import pandas as pd
import numpy as np

from .forecasting.base import _BaseForecaster


__all__ = ['TSCTask', 'ForecastingTask', 'TSCStrategy']
__author__ = ['Markus Löning', 'Sajay Ganesh']


class _BaseTask:
    """An object that encapsulates meta-data and instructions on how to derive the target/label for the time-series
    prediction/supervised learning task.

    Parameters
    ----------
    metadata : pd.DataFrame
        Contains the metadata that the task is expected to work with.
    target : string
        The column header for the target variable to be predicted.
    features : list of string
        The column header for the target variable to be predicted.
        If omitted, every column apart from target would be a feature.
    """
    def __init__(self, target, features=None, metadata=None):
        # TODO input checks on target and feature args
        self._target = target
        self._features = features if features is None else pd.Index(features)

        self._metadata = None  # initialised as None, properly updated through setter method below
        if metadata is not None:
            self.set_metadata(metadata)  # using the modified setter method below

    @property
    def target(self):
        """Expose the private variable _target in a controlled way
        """
        return self._target

    @property
    def features(self):
        """Expose the private variable _features in a controlled way
        """
        return self._features

    @property
    def metadata(self):
        # TODO if metadata is a mutable object itself, its contents may still be mutable
        return self._metadata

    # @metadata.setter
    # def metadata(self, metadata):
    #     if not isinstance(metadata, pd.DataFrame):
    #         raise ValueError(f'Metadata must be provided in form of a pandas dataframe, but found {type(metadata)}')
    #
    #     # only set metadata if metadata is not already set, otherwise raise error
    #     if self._metadata is None:
    #
    #         # complete feature information if not already given
    #         if self.features is None:
    #             self._features = metadata.columns.drop(self.target)
    #
    #             if isinstance(self, TSCTask):
    #                 if self.features is None:
    #                     raise ValueError(f'At least 1 feature column must be given for task of type {type(TSCTask)}')
    #
    #         # otherwise check for consistency against given columns names in metadata
    #         else:
    #             if not np.all(self.features.isin(metadata.columns)):
    #                 raise ValueError(f'Features: {list(self.features)} cannot be found in metadata')
    #
    #         # set metadata
    #         self._metadata = {
    #             "nrow": metadata.shape[0],
    #             "ncol": metadata.shape[1],
    #             "target_type": {self.target: type(i) for i in metadata[self.target]},
    #             "feature_type": {col: {type(i) for i in metadata[col]} for col in self.features}
    #         }
    #
    #     # if metadata is already set, raise error
    #     else:
    #         raise AttributeError('Metadata is already set and can only be set once')

    def set_metadata(self, metadata):
        """Provide metadata to task if not already done so in construction, especially useful in automatic orchestration
        and benchmarking where the metadata is not available in advance.

        Parameters
        ----------
        metadata : pandas DataFrame

        """
        # TODO replace whole pandas data container as input argument with separated metadata container

        if not isinstance(metadata, pd.DataFrame):
            raise ValueError(f'Metadata must be provided in form of a pandas dataframe, but found {type(metadata)}')

        # only set metadata if metadata is not already set, otherwise raise error
        if self._metadata is None:

            # complete feature information if not already given
            if self.features is None:
                self._features = metadata.columns.drop(self.target)

                if isinstance(self, TSCTask):
                    if self.features is None:
                        raise ValueError(f'At least 1 feature column must be given for task of type {type(TSCTask)}')

            # otherwise check for consistency against given columns names in metadata
            else:
                if not np.all(self.features.isin(metadata.columns)):
                    raise ValueError(f'Features: {list(self.features)} cannot be found in metadata')

            # set metadata
            self._metadata = {
                "nrow": metadata.shape[0],
                "ncol": metadata.shape[1],
                "target_type": {self.target: type(i) for i in metadata[self.target]},
                "feature_type": {col: {type(i) for i in metadata[col]} for col in self.features}
            }

        # if metadata is already set, raise error
        else:
            raise AttributeError('Metadata is already set and can only be set once, create a new task for different '
                                 'metadata')

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the task"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]

        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def _get_params(self):
        """Get parameters of the task.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = {key: getattr(self, key, None) for key in self._get_param_names()}
        return out

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self._get_params(), offset=len(class_name), ),)


class TSCTask(_BaseTask):
    """Time series classification task.

    Parameters
    ----------
    metadata : pandas DataFrame
        Meta-data
    target : str
        Name of target variable.
    features : list
        Name of feature variables.
    """
    def __init__(self, target, features=None, metadata=None):
        self._case = 'TSC'
        super(TSCTask, self).__init__(target, features=features, metadata=metadata)


class ForecastingTask(_BaseTask):
    """Forecasting task.

    Parameters
    ----------
    metadata : pandas DataFrame
        Data container
    target : str
        Name of target variable to forecast.
    pred_horizon : list or int
        Single step ahead or list of steps ahead to forecast.


    obs_horizon : str
        - If `fixed`, one or more fixed cut-off points from which to make forecasts
        - If `moving`, iteratively
    features : list
        List of feature variables.
    """
    def __init__(self, target, pred_horizon=None, forecast_type=None, features=None, metadata=None):
        self._case = 'forecasting'

        if isinstance(pred_horizon, list):
            if not np.all([np.issubdtype(type(h), np.integer) for h in pred_horizon]):
                raise ValueError('if pred_horizon is passed as a list, it has to be a list of integers')
        elif np.issubdtype(type(pred_horizon), np.integer) or (pred_horizon is None):
            pass
        else:
            raise ValueError('pred_horizon has to be either a list of integers or single integer')
        self._pred_horizon = 1 if pred_horizon is None else np.sort(pred_horizon)

        if forecast_type not in ('fixed', 'moving'):
            raise ValueError('obs_horizon has to be either `fixed` or `moving`')
        self._obs_horizon = forecast_type

        super(ForecastingTask, self).__init__(target, features=features, metadata=metadata)

    @property
    def pred_horizon(self):
        """Exposes the private variable _pred_horizon in a controlled way
        """
        return self._pred_horizon

    @property
    def obs_horizon(self):
        """Exposes the private variable _pred_horizon in a controlled way
        """
        return self._obs_horizon


class _BaseStrategy:
    """Abstract base strategy class"""
    def __init__(self, check_input=True):
        self._task = None
        self._case = None
        self._traits = {}
        self._is_fitted = False
        self.check_input = check_input

    def __getitem__(self, key):
        """Provide read only access via keys
        to the private traits
        """
        if key not in self._traits.keys():
            raise KeyError
        return self._traits[key]

    def fit(self, task, data):
        """Generic fit method, calls strategy specific `_fit` methods

        Parameters
        ----------
        task : Task
        data : pd.DataFrame

        Returns
        -------
        The fitted strategy
        """

        # check task compatibility with Strategy
        self._check_task_compatibility(task)
        self._task = task

        # update task if necessary
        if self._task.metadata is None:
            self._task.set_metadata(data)

        # strategy-specific implementation
        return self._fit(data)

    def _fit(self, data):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()

    def predict(self, *args):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()

    def _check_task_compatibility(self, task):
        # TODO replace by task-strategy compatibility lookup registry
        if self._case != task._case:
            raise ValueError("Strategy <-> task mismatch: The chosen strategy is incompatible with the given task")


class _BaseSupervisedLearningStrategy(_BaseStrategy):
    """Abstract strategy class for time series supervised learning that accepts a low-level estimator to
    perform a given task.

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of an initialized low-level estimator
    """
    def __init__(self, estimator, check_input=True):
        self._estimator = estimator
        # self._traits = {"tags": None}  # traits for matching strategies with tasks
        super(_BaseSupervisedLearningStrategy, self).__init__(check_input=check_input)

    def _fit(self, data):
        # fit the estimator
        try:
            X = data[self._task.features]
            y = data[self._task.target]
        except KeyError:
            raise ValueError("Task <-> data mismatch. The target/features are not in the data")

        # fit the estimator
        return self._estimator.fit(X, y)

    def predict(self, data):
        """Predict the targets for the test data

        Parameters
        ----------
        data : a pandas DataFrame
            Prediction Data

        Returns
        -------
        predictions: a pd.Dataframe or pd.Series
            returns the predictions
        """
        # predict
        try:
            X = data[self._task.features]
        except KeyError:
            raise ValueError("Task <-> data mismatch. The target/features are not in the data")

        # estimate predictions and return
        predictions = self._estimator.predict(X)
        return predictions

    def get_params(self, deep=True):
        """call get_params of the estimator
        """
        return self._estimator.get_params(deep=deep)

    def set_params(self, **params):
        """Call set_params of the estimator
        """
        self._estimator.set_params(**params)

    def __repr__(self):
        strategy_name = self.__class__.__name__
        estimator_name = self._estimator.__class__.__name__
        return '%s(%s(%s))' % (strategy_name, estimator_name,
                               _pprint(self.get_params(deep=False), offset=len(strategy_name), ),)


class TSCStrategy(_BaseSupervisedLearningStrategy):
    """Strategy for time series classification
    """
    def __init__(self, estimator):
        super(TSCStrategy, self).__init__(estimator)
        self._case = "TSC"


class _BaseForecastingStrategy(_BaseStrategy, _BaseForecaster):
    """Abstract class for forecasting strategies
    """
    def __init__(self, check_input=True):
        super(_BaseForecastingStrategy, self).__init__(check_input=check_input)
        self._case = 'forecasting'

        self._is_updated = False
        self._target_idx = None
        self._estimator = None
        self._fitted_estimator = None
        self._updated_estimator = None

    def _fit(self, data):
        if self.check_input:
            self._check_fit_data(data)

        # store index of target variable to be predicted
        target = data[self._task.target].iloc[0]
        self._target_idx = target.index if hasattr(target, 'index') else pd.RangeIndex(len(target))

        self._fit_strategy(data)  # strategy specific implementation
        self._is_fitted = True
        return self

    def update(self, data):
        check_is_fitted(self, '_is_fitted')
        if self.check_input:
            self._check_update_data(data)

        data = self._transform_data(data)
        self._update(data)  # strategy specific implementation
        self._is_updated = True
        return self

    def predict(self):
        check_is_fitted(self, '_is_fitted')
        return self._predict()  # forecaster specific implementation

    def _update(self, data):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()

    def _fit_strategy(self, data):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()

    @staticmethod
    def _check_fit_data(data):
        # TODO input checks for forecasting
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be supplied as a pandas DataFrame, but found {type(data)}')
        if not data.shape[0] == 1:
            raise ValueError(f'Data must be from a single instance (row), but found {data.shape[0]} rows')

    def _check_update_data(self, data):
        # TODO input checks for forecasting
        target = data[self._task.target].iloc[0]
        updated_target_idx = target.index if hasattr(target, 'index') else pd.RangeIndex(len(target))
        is_after = updated_target_idx.min() == self._target_idx.max()
        is_same_type = isinstance(updated_target_idx, type(self._target_idx))
        if not (is_after and is_same_type):
            raise ValueError('Data passed to `update` does not match the data passed to `fit`')

    def _predict(self):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()

    def _transform_data(self, data):
        """Helper function to transform nested data with series/arrays in cells into pd.Series with primitives in cells
        """
        return pd.Series(*data[self._task.target].tolist())

    def __repr__(self):
        strategy_name = self.__class__.__name__
        return '%s(%s)' % (strategy_name, _pprint(self.get_params(deep=False), offset=len(strategy_name), ),)


class _SingleSeriesForecastingStrategy(_BaseForecastingStrategy):
    """Classical forecaster which implements predict method for single-series/univariate fitted/updated classical
    forecasting techniques without exogenous variables.
    """

    def _predict(self):

        if self._task.pred_horizon == 'fixed':
            # Convert step-ahead prediction horizon into zero-based index
            pred_horizon = self._task.pred_horizon
            pred_horizon_idx = pred_horizon - np.min(pred_horizon)

            if self._is_updated:
                # Predict updated (pre-initialised) model with start and end values relative to end of train series
                start = self._task.pred_horizon[0]
                end = self._task.pred_horizon[-1]
                pred = self._updated_estimator.predict(start=start, end=end)

            else:
                # Predict fitted model with start and end points relative to start of train series
                pred_horizon = pred_horizon + len(self._target_idx) - 1
                start = pred_horizon[0]
                end = pred_horizon[-1]
                pred = self._fitted_estimator.predict(start=start, end=end)

            # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
            return pred.iloc[pred_horizon_idx]

        elif self._task.obs_horizon == 'moving':
            raise NotImplementedError()

        else:
            # TODO necessary to check this here or can we safely assume obs_horizon is always either fixed or moving
            raise ValueError('obs_horizon of task ')

    def _update(self, data):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()

    def _fit_strategy(self, data):
        # TODO only works on univariate/single series forecasting without exogenous variables
        data = self._transform_data(data)
        return self._fit_estimator(data)  # estimator specific implementations

    def _fit_estimator(self, data):
        """Placeholder to be overwritten by specific strategies"""
        raise NotImplementedError()
