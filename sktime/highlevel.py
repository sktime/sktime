"""
Unified high-level interface for various time series related learning tasks.
"""

import numpy as np
import pandas as pd
from sklearn.base import _pprint
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from inspect import signature

from .classifiers.base import BaseClassifier
from .forecasting.base import BaseForecaster
from .regressors.base import BaseRegressor
from sktime.utils.transformations import RollingWindowSplit
from .utils.validation import validate_fh

__all__ = ['TSCTask',
           'TSRTask',
           'ForecastingTask',
           'TSCStrategy',
           'TSRStrategy',
           'ForecastingStrategy']
__author__ = ['Markus Löning', 'Sajay Ganesh']


# TODO implement task-strategy-estimator compatibility lookup registry using strategy traits
REGRESSOR_TYPES = (BaseRegressor, RegressorMixin)
CLASSIFIER_TYPES = (BaseClassifier, ClassifierMixin)
FORECASTER_TYPES = (BaseForecaster, )
ESTIMATOR_TYPES = REGRESSOR_TYPES + CLASSIFIER_TYPES + FORECASTER_TYPES

CASES = ("TSR", "TSC", "Forecasting")


class BaseTask:
    """
    Abstract base task class.

    A task encapsulates metadata information such as the feature and
    target variable which to fit the data to and additional necessary
    instructions on how to fit and predict.

    Implements attributes and operations shared by all tasks,
    including compatibility checks between the concrete task type and
    passed metadata.

    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optinal, (default=None)
        The column name(s) for the feature variable. If None, every column apart from target will be used as a feature.
    metadata : pandas.DataFrame
        Contains the metadata that the task is expected to work with.
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
        """
        Make attributes read-only.
        """
        return self._target

    @property
    def features(self):
        """
        Make attributes read-only.
        """
        return self._features

    @property
    def metadata(self):
        """
        Make attributes read-only.
        """
        # TODO if metadata is a mutable object itself, its contents may still be mutable
        return self._metadata

    def set_metadata(self, metadata):
        """
        Provide missing metadata information to task if not already set.

        This method is especially useful in orchestration where metadata may not be
        available when specifying the task.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Metadata container

        Returns
        -------
        self : an instance of self
        """
        # TODO replace whole pandas data container as input argument with separated metadata container

        # only set metadata if metadata is not already set, otherwise raise error
        if self._metadata is not None:
            raise AttributeError('Metadata is already set and can only be set once, create a new task for different '
                                 'metadata')

        # check for consistency of information provided in task with given metadata
        self.check_data_compatibility(metadata)

        # set default feature information (all columns but target) using metadata
        if self.features is None:
            self._features = metadata.columns.drop(self.target)

        # set metadata
        self._metadata = {
            "nrow": metadata.shape[0],
            "ncol": metadata.shape[1],
            "target_type": {self.target: type(i) for i in metadata[self.target]},
            "feature_type": {col: {type(i) for i in metadata[col]} for col in self.features}
        }
        return self

    def check_data_compatibility(self, metadata):
        """
        Check compatibility of task with passed metadata.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Metadata container
        """
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError(f'Metadata must be provided in form of a pandas dataframe, but found: {type(metadata)}')

        if self.target not in metadata.columns:
            raise ValueError(f"Target: {self.target} not found in metadata")

        if self.features is not None:
            if not np.all(self.features.isin(metadata.columns)):
                raise ValueError(f"Features: {list(self.features)} not found in metadata")

        if isinstance(self, (TSCTask, TSRTask)):
            if self.features is None:
                if len(metadata.columns.drop(self.target)) == 0:
                    raise ValueError(f"For task of type: {type(self)}, at least one feature must be given, "
                                     f"but found none")

            if metadata.shape[0] <= 1:
                raise ValueError(f"For task of type: {type(self)}, several samples (rows) must be given, but only "
                                 f"found: {metadata.shape[0]} samples")

        if isinstance(self, ForecastingTask):
            if metadata.shape[0] > 1:
                raise ValueError(f"For task of type: {type(self)}, only a single sample (row) can be given, but found: "
                                 f"{metadata.shape[0]} rows")

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
    """
    Time series classification task.

    A task encapsulates metadata information such as the feature and target variable
    to which to fit the data to and any additional necessary instructions on how
    to fit and predict.

    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optinal (default=None)
        The column name(s) for the feature variable. If None, every column apart from target will be used as a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        self._case = 'TSC'
        super(TSCTask, self).__init__(target, features=features, metadata=metadata)


class TSRTask(BaseTask):
    """
    Time series regression task.

    A task encapsulates metadata information such as the feature and target variable
    to which to fit the data to and any additional necessary instructions on how
    to fit and predict.

    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optinal (default=None)
        The column name(s) for the feature variable. If None, every column apart from target will be used as a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        self._case = 'TSR'
        super(TSRTask, self).__init__(target, features=features, metadata=metadata)


class ForecastingTask(BaseTask):
    """
    Time series forecasting task.

    A task encapsulates metadata information such as the feature and target
    variable which to fit the data to and additional necessary instructions on how to fit and predict.


    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optional, (default=None)
        The column name(s) for the exogenous feature variable. If None, every column apart from target will be used as
        a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.
    """

    def __init__(self, target, fh=None, features=None, metadata=None):
        self._case = "Forecasting"
        self._fh = validate_fh(fh)
        super(ForecastingTask, self).__init__(target, features=features, metadata=metadata)

    @property
    def fh(self):
        """
        Makes attribute read-only.
        """
        return self._fh


class BaseStrategy:
    """
    Abstract base strategy class.

    Implements attributes and operations shared by all strategies,
    including input and compatibility checks between passed estimator,
    data and task.
    """

    def __init__(self, estimator, name=None, check_input=True):
        self._check_estimator_compatibility(estimator)

        self._estimator = estimator

        self._name = estimator.__class__.__name__ if name is None else name
        self.check_input = check_input
        self._task = None

    @property
    def name(self):
        """
        Makes attribute read-only.
        """
        return self._name

    def __getitem__(self, key):
        """
        Provide read only access via keys to the private traits
        """
        if key not in self._traits.keys():
            raise KeyError
        return self._traits[key]

    def fit(self, task, data):
        """
        Fit the strategy to the given task and data.

        Parameters
        ----------
        task : Task
            Task encapsualting metadata information on feature and target variables to which to fit the data to.
        data : pandas.DataFrame
            Dataframe with feature and target variables as specified in task.

        Returns
        -------
        self : an instance of the self
        """
        if self.check_input:
            self._validate_data(data)

        # Check task compatibility with strategy
        self._check_task_compatibility(task)
        self._task = task

        # Set metadata if not already set
        if self._task.metadata is None:
            self._task.set_metadata(data)

        # strategy-specific implementation
        return self._fit(data)

    def _check_task_compatibility(self, task):
        """
        Check compatibility of task with strategy
        """
        # TODO replace by task-strategy compatibility lookup registry
        if hasattr(task, '_case'):
            if self._case != task._case:
                raise ValueError("Strategy <-> task mismatch: The chosen strategy is incompatible with the given task")
        else:
            raise AttributeError("The passed case of the task is unknown")

    def _check_estimator_compatibility(self, estimator):
        """
        Check compatibility of estimator with strategy
        """

        # Determine required estimator type from strategy case
        # TODO replace with strategy - estimator type registry lookup
        if hasattr(self, '_traits'):
            required = self._traits["required_estimator_type"]
            if any(estimator_type not in ESTIMATOR_TYPES for estimator_type in required):
                raise AttributeError(f"Required estimator type unknown")
        else:
            raise AttributeError(f"Required estimator type not found")

        # Check estimator compatibility with required type
        if not isinstance(estimator, BaseEstimator):
            raise ValueError(f"Estimator must inherit from BaseEstimator")

        # If pipeline, check compatibility of final estimator
        if isinstance(estimator, Pipeline):
            final_estimator = estimator.steps[-1][1]
            if not isinstance(final_estimator, required):
                raise ValueError(f"Final estimator of passed pipeline estimator must be of type: {required}, "
                                 f"but found: {type(final_estimator)}")

        # If tuning meta-estimator, check compatibility of inner estimator
        elif isinstance(estimator, (GridSearchCV, RandomizedSearchCV)):
            estimator = estimator.estimator
            if not isinstance(estimator, required):
                raise ValueError(f"Inner estimator of passed meta-estimator must be of type: {required}, "
                                 f"but found: {type(estimator)}")

        # Otherwise check estimator directly
        else:
            if not isinstance(estimator, required):
                raise ValueError(f"Passed estimator has to be of type: {required}, but found: {type(estimator)}")

    @staticmethod
    def _validate_data(data):
        """
        Helper function to validate input data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be pandas DataFrame, but found: {type(data)}")

        # TODO add input checks for contents, ie all cells be pandas Series, numpy arrays or primitives,
        #  ultimately move checks to data container
        # s = y.iloc[0]
        # if not isinstance(s, (np.ndarray, pd.Series)):
        #     raise ValueError(f'``y`` must contain a pandas Series or numpy array, but found: {type(s)}.')

    def get_params(self, deep=True):
        """
        Call get_params of the estimator. Retrieves hyper-parameters.

        Returns
        -------
        params : dict
            Dictionary with parameter names and values of estimator.
        """
        return self._estimator.get_params(deep=deep)

    def set_params(self, **params):
        """
        Call set_params of the estimator. Sets hyper-parameters.
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

    Implements predict and internal fit methods for time series regression and classification.
    """

    def _fit(self, data):
        """
        Internal fit

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with feature and target variables as specified in task.


        Returns
        -------
        self : an instance of self
        """
        # select features and target
        X = data[self._task.features]
        y = data[self._task.target]

        # fit the estimator
        return self._estimator.fit(X, y)

    def predict(self, data):
        """
        Predict using the given test data.

        Parameters
        ----------
        data : a pandas.DataFrame
            Dataframe with feature and target variables as specified in task passed to ``fit``.


        Returns
        -------
        y_pred : pandas.Series
            Returns the series of predicted values.
        """

        # select features
        X = data[self._task.features]

        # predict
        return self._estimator.predict(X)


class TSCStrategy(BaseSupervisedLearningStrategy):
    """
    Strategy for time series classification.

    Parameters
    ----------
    estimator : an estimator
        Low-level estimator used in strategy.
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """

    def __init__(self, estimator, name=None, check_input=True):
        self._case = "TSC"
        self._traits = {"required_estimator_type": CLASSIFIER_TYPES}
        super(TSCStrategy, self).__init__(estimator, name=name, check_input=check_input)


class TSRStrategy(BaseSupervisedLearningStrategy):
    """
    Strategy for time series regression.

    Parameters
    ----------
    estimator : an estimator
        Low-level estimator used in strategy.
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """

    def __init__(self, estimator, name=None, check_input=True):
        self._case = "TSR"
        self._traits = {"required_estimator_type": REGRESSOR_TYPES}
        super(TSRStrategy, self).__init__(estimator, name=name, check_input=check_input)


class ForecastingStrategy(BaseStrategy):
    """
    Strategy for time series forecasting.

    Parameters
    ----------
    estimator : an estimator
        Low-level estimator
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """

    def __init__(self, estimator, name=None, check_input=True):
        self._case = "Forecasting"
        self._traits = {"required_estimator_type": FORECASTER_TYPES}
        super(ForecastingStrategy, self).__init__(estimator, name=name, check_input=check_input)

    def _fit(self, data):
        """
        Internal fit.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data

        Returns
        -------
        self : an instance of self
        """

        y = data[self._task.target]
        if len(self._task.features) > 0:
            X = data[self._task.features]
            kwargs = {'X': X}
        else:
            kwargs = {}

        # fit the estimator
        return self._estimator.fit(y, **kwargs)

    def update(self, data):
        """
        Update forecasts using new data.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with feature and target variables as specified in task.

        Returns
        -------
        self : an instance of the self
        """

        if self.check_input:
            self._task.check_data_compatibility(data)

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
        """
        Predict.

        Parameters
        ----------
        data : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that if
            provided, the forecaster must also have been fitted on the exogenous
            features.

        Returns
        -------
        y_pred : pandas.Series
            Series of predicted values.
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


class Forecasting2TSRReductionStrategy(BaseStrategy):
    """
    Forecasting to time series regression reduction strategy.

    Strategy to reduce a forecasting problem to a time series regression
    problem using a rolling window approach

    Parameters
    ----------
    estimator : an estimator
        Time series regressor.
    window_length : int, optional (default=None)
        Window length of rolling window approach.
    name : str, optional (default=None)
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    def __init__(self, estimator, window_length=None, name=None, check_input=True):
        self._case = "Forecasting"
        self._traits = {"required_estimator_type": REGRESSOR_TYPES}
        super(Forecasting2TSRReductionStrategy, self).__init__(estimator, name=name, check_input=check_input)

        # TODO what's a good default for window length? sqrt(len(data))?
        self.window_length = window_length
        self.estimators = []
        self.estimators_ = []

    def _fit(self, data):
        """
        Internal fit.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data

        Returns
        -------
        self : an instance of self
        """

        # Select target and feature variables
        y = data[self._task.target]
        if len(self._task.features) > 0:
            X = data[self._task.features]
            # TODO how to handle exogenous variables
            raise NotImplementedError()

        # Set up window roller
        fh = self._task.fh
        rw = RollingWindowSplit(window_length=self.window_length, fh=fh)
        self.rw = rw

        # Unnest target series
        yt = y.iloc[0]
        index = np.arange(len(yt))

        # Transform target series into tabular format using rolling window splits
        xs = []
        ys = []
        for feature_window, target_window in rw.split(index):
            x = yt[feature_window]
            y = yt[target_window]
            xs.append(x)
            ys.append(y)

        # Construct nested pandas DataFrame for X
        X = pd.DataFrame(pd.Series([x for x in np.array(xs)]))
        Y = np.array(ys)

        # Clone estimators, one for each step in the forecasting horizon
        n_steps = len(fh)
        self.estimators = [clone(self._estimator) for _ in range(n_steps)]

        # Iterate over estimators/forecast horizon
        for estimator, y in zip(self.estimators, Y.T):
            y = pd.Series(y)
            estimator.fit(X, y)
            self.estimators_.append(estimator)

        # Save the last window-length number of observations for predicting
        self.window_length_ = rw.get_window_length()
        self._last_window = yt.iloc[-self.window_length_:]

        return self

    def predict(self, data=None):
        """
        Predict.

        Parameters
        ----------
        data : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that if
            provided, the forecaster must also have been fitted on the exogenous
            features.

        Returns
        -------
        y_pred : pandas.Series
            Series of predicted values.
        """

        fh = self._task.fh

        if data is not None:
            # TODO handle exog data
            raise NotImplementedError()

        # Predict using last window (single row) and fitted estimators
        x = pd.DataFrame(pd.Series([self._last_window]))
        y_pred = np.zeros(len(fh))

        # Iterate over estimators/forecast horizon
        for i, estimator in enumerate(self.estimators_):
            y_pred[i] = estimator.predict(x)

        # Add name and predicted index
        index = self._last_window.index[-1] + fh
        name = self._last_window.name
        return pd.Series(y_pred, name=name, index=index)
