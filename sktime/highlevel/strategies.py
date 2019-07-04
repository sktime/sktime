"""
Unified high-level interface for various time series related learning strategies.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import _pprint
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sktime.utils.time_series import RollingWindowSplit
from sktime.classifiers.base import BaseClassifier
from sktime.forecasters.base import BaseForecaster
from sktime.regressors.base import BaseRegressor
from sktime.utils.validation import validate_fh

__all__ = ["TSCStrategy", "TSRStrategy", "ForecastingStrategy", "Forecasting2TSRReductionStrategy"]
__author__ = ['Markus Löning', 'Sajay Ganesh']


# TODO implement task-strategy-estimator compatibility lookup registry using strategy traits
REGRESSOR_TYPES = (BaseRegressor, RegressorMixin)
CLASSIFIER_TYPES = (BaseClassifier, ClassifierMixin)
FORECASTER_TYPES = (BaseForecaster, )
ESTIMATOR_TYPES = REGRESSOR_TYPES + CLASSIFIER_TYPES + FORECASTER_TYPES

CASES = ("TSR", "TSC", "Forecasting")


class BaseStrategy(BaseEstimator):
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
        """Makes attribute accessible, but read-only.
        """
        return self._name

    @property
    def estimator(self):
        """Makes attribute accessible, but read-only.
        """
        return self._estimator

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

    def save(self, dataset_name, cv_fold, strategies_save_dir):
        """
        Saves the strategy on the hard drive
        Parameters
        ----------
        dataset_name:string
            Name of the dataset
        cv_fold: int
            Number of cross validation fold on which the strategy was trained
        strategies_save_dir: string
            Path were the strategies will be saved
        """
        if strategies_save_dir is None:
            raise ValueError('Please provide a directory for saving the strategies')

        # TODO implement check for overwriting already saved files
        save_path = os.path.join(strategies_save_dir, dataset_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # TODO pickling will not work for all strategies
        pickle.dump(self, open(os.path.join(save_path, self.name + '_cv_fold' + str(cv_fold) + '.p'), "wb"))

    def load(self, path):
        """
        Load saved strategy
        Parameters
        ----------
        path: String
            location on disk where the strategy was saved

        Returns
        -------
        strategy:
            sktime strategy
        """
        return pickle.load(open(path, 'rb'))

    def __repr__(self):
        strategy_name = self.__class__.__name__
        estimator_name = self.estimator.__class__.__name__
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
        return self.estimator.fit(X, y)

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
        return self.estimator.predict(X)


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
    Strategy for time series forecasters.

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
        fh = self._task.fh

        if len(self._task.features) > 0:
            X = data[self._task.features]
            kwargs = {'X': X}
        else:
            kwargs = {}

        # fit the estimator
        return self.estimator.fit(y, fh=fh, **kwargs)

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

        if hasattr(self.estimator, 'update'):
            try:
                y = data[self._task.target]
                if len(self._task.features) > 0:
                    X = data[self._task.features]
                    kwargs = {'X': X}
                else:
                    kwargs = {}
            except KeyError:
                raise ValueError("Task <-> data mismatch. The target/features are not in the data")
            self.estimator.update(y, **kwargs)
        else:
            raise NotImplementedError(f"Supplied low-level estimator: {self.estimator} does not implement update "
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

        return self.estimator.predict(fh=fh, **kwargs)  # forecaster specific implementation


class Forecasting2TSRReductionStrategy(BaseStrategy):
    """
    Forecasting to time series regression reduction strategy.

    Strategy to reduce a forecasters problem to a time series regression
    problem using a rolling window approach

    Parameters
    ----------
    estimator : an estimator
        Time series regressor.
    window_length : int, optional (default=None)
        Window length of rolling window approach.
    dynamic : bool, optional (default=False)
        - If True, estimator is fitted for one-step ahead forecasts and only one-step ahead forecasts are made using
        extending the last window of the training data with already made forecasts.
        - If False, one estimator is fitted for each step-ahead forecast and only the last window is used for making
        forecasts.
    name : str, optional (default=None)
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    def __init__(self, estimator, window_length=None, dynamic=False, name=None, check_input=True):
        self._case = "Forecasting"
        self._traits = {"required_estimator_type": REGRESSOR_TYPES}
        super(Forecasting2TSRReductionStrategy, self).__init__(estimator, name=name, check_input=check_input)

        # TODO what's a good default for window length? sqrt(len(data))?
        self.window_length = window_length
        self.dynamic = dynamic

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
        # For dynamic prediction, models are only trained on one-step ahead forecast
        fh = 1 if self.dynamic else self._task.fh
        fh = validate_fh(fh)
        n_fh = len(fh)

        self.rw = RollingWindowSplit(window_length=self.window_length, fh=fh)

        # Unnest target series
        yt = y.iloc[0]
        index = np.arange(len(yt))

        # Transform target series into tabular format using rolling window splits
        xs = []
        ys = []
        for feature_window, target_window in self.rw.split(index):
            x = yt[feature_window]
            y = yt[target_window]
            xs.append(x)
            ys.append(y)

        # Construct nested pandas DataFrame for X
        X = pd.DataFrame(pd.Series([x for x in np.array(xs)]))
        Y = np.array([np.array(y) for y in ys])

        # Fitting
        if self.dynamic:
            # Fit estimator for one-step ahead forecast
            y = Y.ravel()  # convert into one-dimensional array
            estimator = clone(self.estimator)
            estimator.fit(X, y)
            self.estimator_ = estimator

        else:
            # Fit one estimator for each step-ahead forecast
            self.estimators = []
            self.estimators_ = []

            n_fh = len(fh)

            # Clone estimators
            self.estimators = [clone(self.estimator) for _ in range(n_fh)]

            # Iterate over estimators/forecast horizon
            for estimator, y in zip(self.estimators, Y.T):
                y = pd.Series(y)
                estimator.fit(X, y)
                self.estimators_.append(estimator)

        # Save the last window-length number of observations for predicting
        self.window_length_ = self.rw.get_window_length()
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
        if data is not None:
            # TODO handle exog data
            raise NotImplementedError()

        # get forecasting horizon
        fh = self._task.fh
        n_fh = len(fh)

        # use last window as test data for prediction
        x_test = pd.DataFrame(pd.Series([self._last_window]))
        y_pred = np.zeros(len(fh))

        # prediction can be either dynamic making only one-step ahead forecasts using previous forecasts or static using
        # only the last window and using one fitted estimator for each step ahead forecast
        if self.dynamic:
            # Roll/extend last window using previous one-step ahead forecasts
            for i in range(n_fh):
                y_pred[i] = self.estimator_.predict(x_test)

                # append prediction to last window and update x
                x_test = np.append(x_test.iloc[0, 0].values, y_pred[i])[-self.window_length_:]

                # put data into required format
                x_test = pd.DataFrame(pd.Series([pd.Series(x_test)]))

        else:
            # Iterate over estimators/forecast horizon
            for i, estimator in enumerate(self.estimators_):
                y_pred[i] = estimator.predict(x_test)

        # Add name and forecast index
        index = self._last_window.index[-1] + fh
        name = self._last_window.name

        return pd.Series(y_pred, name=name, index=index)
