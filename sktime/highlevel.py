"""
Unified high-level interface for various time series related learning tasks.
"""

from sklearn.base import _pprint
from sklearn.utils.validation import check_is_fitted
from inspect import signature
import pandas as pd
import numpy as np

from .forecasting.base import BaseForecaster
from .classifiers.base import BaseClassifier
from .regressors.base import BaseRegressor

__all__ = ['TSCTask', 'ForecastingTask', 'TSCStrategy']
__author__ = ['Markus Löning', 'Sajay Ganesh']

# TODO implement compatibility checks between metadata and task: e.g. for tsc, if features are present
# TODO implement task-strategy compatibility lookup registry using strategy traits


class BaseTask:
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
        if self._metadata is not None:
            raise AttributeError('Metadata is already set and can only be set once, create a new task for different '
                                 'metadata')

        else:
            # set default feature information (all columns but target) using metadata
            if self.features is None:
                self._features = metadata.columns.drop(self.target)

            # check for consistency of task with given metadata
            self.check_compatibility_with_data(metadata)

            # set metadata
            self._metadata = {
                "nrow": metadata.shape[0],
                "ncol": metadata.shape[1],
                "target_type": {self.target: type(i) for i in metadata[self.target]},
                "feature_type": {col: {type(i) for i in metadata[col]} for col in self.features}
            }

    def check_compatibility_with_data(self, metadata):
        """Helper function to check compatibility of task with data"""
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError(f'Metadata must be provided in form of a pandas dataframe, but found: {type(metadata)}')

        if not np.all(self.features.isin(metadata.columns)):
            raise ValueError(f"Features: {list(self.features)} not found in metadata")

        if self.target not in metadata.columns:
            raise ValueError(f"Target: {self.target} not found in metadata")

        if isinstance(self, ForecastingTask):
            if metadata.shape[0] > 1:
                raise ValueError(f"For task of type: {type(self)}, only a single sample (row) can be given, but found: "
                                 f"{metadata.shape[0]} rows")

        if isinstance(self, (TSCTask, TSRTask)):
            if len(self.features) == 0:
                raise ValueError(f"For task of type: {type(self)}, at least one feature must be given, but found none")

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


class TSCTask(BaseTask):
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


class TSRTask(BaseTask):
    """Time series regression task.

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
        self._case = 'TSR'
        super(TSRTask, self).__init__(target, features=features, metadata=metadata)


class ForecastingTask(BaseTask):
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
    def __init__(self, target, fh=None, features=None, metadata=None):
        self._case = 'forecasting'

        if isinstance(fh, list):
            if not np.all([np.issubdtype(type(h), np.integer) for h in fh]):
                raise ValueError('if pred_horizon is passed as a list, it has to be a list of integers')
        elif np.issubdtype(type(fh), np.integer) or (fh is None):
            pass
        else:
            raise ValueError('pred_horizon has to be either a list of integers or single integer')
        self._fh = 1 if fh is None else np.sort(fh)

        super(ForecastingTask, self).__init__(target, features=features, metadata=metadata)

    @property
    def fh(self):
        """Exposes the private variable forecast horizon (fh) in a controlled way
        """
        return self._fh


class BaseStrategy:
    """Abstract base strategy class"""
    def __init__(self, estimator, name=None, check_input=True):
        self._name = estimator.__class__.__name__ if name is None else name
        self._estimator = estimator
        self._task = None
        self._case = None
        self._traits = {}
        self.check_input = check_input

    @property
    def name(self):
        return self._name

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

    def _check_task_compatibility(self, task):
        # TODO replace by task-strategy compatibility lookup registry
        if self._case != task._case:
            raise ValueError("Strategy <-> task mismatch: The chosen strategy is incompatible with the given task")

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


class BaseSupervisedLearningStrategy(BaseStrategy):
    """Abstract strategy class for time series supervised learning that accepts a low-level estimator to
    perform a given task.

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of an initialized low-level estimator
    """
    def _fit(self, data):
        # fit the estimator
        try:
            X = data[self._task.features]
            y = data[self._task.target]
        except KeyError:
            raise ValueError("Task <-> data mismatch. The target/features are not in the data")

        # fit the estimator
        return self._estimator.fit(X, y)

    def predict(self, data=None):
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
        y_pred = self._estimator.predict(X)
        return y_pred


class TSCStrategy(BaseSupervisedLearningStrategy):
    """Strategy for time series classification
    """
    def __init__(self, estimator, name=None, check_input=True):
        if not isinstance(estimator, BaseClassifier):
            raise ValueError(f"Passed estimator must be a classifier, but found: {type(estimator)}")
        super(TSCStrategy, self).__init__(estimator, name=name, check_input=check_input)
        self._case = "TSC"


class TSRStrategy(BaseSupervisedLearningStrategy):
    """Strategy for time series regression
    """
    def __init__(self, estimator, name=None, check_input=True):
        if not isinstance(estimator, BaseRegressor):
            raise ValueError(f"Passed estimator must be a regressor, but found: {type(estimator)}")
        super(TSRStrategy, self).__init__(estimator, name=name, check_input=check_input)
        self._case = "TSR"


class ForecastingStrategy(BaseStrategy):
    """Abstract class for forecasting strategies
    """
    def __init__(self, estimator, name=None, check_input=True):
        if not isinstance(estimator, BaseForecaster):
            raise ValueError(f"Passed estimator must be a forecaster, but found: {type(estimator)}")
        super(ForecastingStrategy, self).__init__(estimator, name=name, check_input=check_input)
        self._case = 'forecasting'

    def _fit(self, data):
        try:
            y = data[self._task.target]
            if len(self._task.features) > 0:
                X = data[self._task.features]
                kwargs = {'X': X}
            else:
                kwargs = {}
        except KeyError:
            raise ValueError("Task <-> data mismatch. The target/features are not in the data")

        # fit the estimator
        self._estimator.fit(y, **kwargs)
        self._is_fitted = True
        return self

    def update(self, data):
        """Update forecasts using new data.

        Parameters
        ----------
        data

        Returns
        -------

        """
        if self.check_input:
            self._task.check_compatibility_with_data(data)
            self._check_update_data(data)

        if hasattr(self._estimator, 'update'):
            try:
                y = data[self._task.target]
                if len(self._task.features) > 0:
                    X = data[self._task.features]
                    kwargs = {'X': X}
                else:
                    kwargs = {}
            except KeyError:
                raise ValueError("Task <-> data mismatch. The target/features are not in the data")
            self._estimator.update(y, **kwargs)
        else:
            raise NotImplementedError(f"Supplied low-level estimator: {self._estimator} does not implement update "
                                      f"method.")

        return self

    def predict(self, data=None):
        """Make predictions

        Parameters
        ----------
        data

        Returns
        -------

        """
        fh = self._task.fh

        if len(self._task.features) > 0:
            if data is not None:
                X = data[self._task.features]
                kwargs = {'X': X}
            else:
                raise ValueError('No data passed, but passed task requires feature data')
        else:
            kwargs = {}

        return self._estimator.predict(fh=fh, **kwargs)  # forecaster specific implementation

    @staticmethod
    def _check_fit_data(data):
        """Helper function to check input data for fit"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be supplied as a pandas DataFrame, but found {type(data)}')
        if not data.shape[0] == 1:
            raise ValueError(f'Data must be from a single instance (row), but found {data.shape[0]} rows')

    def _check_update_data(self, data):
        """Helper function to check input data for update"""
        # TODO add additional input checks for update data
        y = data[self._task.target].iloc[0]
        y_updated_idx = y.index if hasattr(y, 'index') else pd.RangeIndex(len(y))
        is_same_type = isinstance(y_updated_idx, type(self._estimator._y_idx))
        if not is_same_type:
            raise ValueError('Data passed to `update` does not match the data passed to `fit`')
