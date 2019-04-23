"""
Unified high-level interface for various time series related learning tasks.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import _pprint

__all__ = ['TSCTask', 'ForecastingTask', 'TSCStrategy', 'TSRStrategy']


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
<<<<<<< HEAD
    def __init__(self, case=None, data=None, target=None, features=None):
        # check if all necessary keyword arguments are present
        # if not(case and data and target):
        #     raise ValueError("All three kerword arguments case, data and target should be supplied")
        self._case = case
        self._target = target
        # by default every column apart from target is a feature
        if features is None:
            data = data.load()
            self._features = data.columns.drop(self._target)
        else:
            # set the user-supplied feature list as read-only
            self._features = pd.Index(features)
=======
    def __init__(self, target, features=None, metadata=None):
>>>>>>> dev

        # TODO input checks
        # TODO use metadata object from extended data container
        self._target = target
        self._features = features if features is None else pd.Index(features)
        self._metadata = None

        if metadata is not None:
            self.set_metadata(metadata)

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

    def __getitem__(self, key):
        """Provide read only access via keys to the metadata of the task
        """
        if key not in self._metadata.keys():
            raise KeyError()
        return self._metadata[key]

    def set_metadata(self, metadata):

        if not isinstance(metadata, pd.DataFrame):
            # TODO replace whole pandas datacontainer with separated metadata container
            raise ValueError(f'Data must be pandas dataframe, but found {type(metadata)}')

        # if metadata is not already set
        if self._metadata is None:
            # update features if not already set
            if self.features is None:
                self._features = metadata.columns.drop(self.target)
            # otherwise check against metadata columns
            else:
                if not np.all(self.features.isin(metadata.columns)):
                    raise ValueError(f'Features: {self.features} cannot be found in metadata')

            self._metadata = {
                "nrow": metadata.shape[0],
                "ncol": metadata.shape[1],
                "target_type": {self.target: type(i) for i in metadata[self.target]},
                "feature_type": {col: {type(i) for i in metadata[col]} for col in self.features}
            }

        # if metadata is already set, raise error
        else:
            raise AttributeError('Metadata is already set and can only be set once')


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


class ForecastingTask(BaseTask):
    """Forecasting task.

    Parameters
    ----------
    metadata : pandas DataFrame
        Meta-metadata
    target : str
        Name of target variable.
    pred_horizon : list
        List of steps ahead to predict.
    features : list
        List of feature variables.
    """
    def __init__(self, target, pred_horizon, features=None, metadata=None):
        self._case = 'forecasting'
        self._pred_horizon = np.sort(pred_horizon)
        super(ForecastingTask, self).__init__(target, features=features, metadata=metadata)

    @property
    def pred_horizon(self):
        """Exposes the private variable _pred_horizon in a controlled way
        """
        return self._pred_horizon

class BaseStrategy:
    """A meta-estimator that employs a low level estimator to
    perform a pescribed task

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of an appropriately initialized
        low-level estimator
    """
    def __init__(self, estimator):
        # construct and initialize the estimator
        self._estimator = estimator
        self._case = None
        self._task = None
<<<<<<< HEAD
        self._meta = {"tags": None}
        self._name = estimator.__class__.__name__ #TODO: sets the name of the estimator. Not sure if this works for estimators other than scikit learn
=======
        # self._traits = {"tags": None}  # traits for matching strategies with tasks
>>>>>>> dev

    @property
    def case(self):
        """Expose the private variable _case as read only
        """
        return self._case
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
        """Fit the estimator according to task details

        Parameters
        ----------
        task : Task
            A task initialized with the same kind of data
        data : pd.DataFrame
            Training Data

        Returns
        -------
        self: the instance being fitted
            returns the predictions
        """
        # check task compatibility with Strategy
        # TODO replace by task-strategy compatibility lookup registry
        if self._case != task._case:
            raise ValueError("Strategy <-> task mismatch: The chosen strategy is incompatible with the given task")

        # update task if necessary
        if task.features is None:
            task.set_metadata(data)
        self._task = task

        # fit the estimator
        try:
            X = data[self._task.features]
            y = data[self._task.target]
        except KeyError:
            raise ValueError("Task <-> data mismatch. The target/features are not in the data")

        # fit the estimator
        self._estimator.fit(X, y)
        return self

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
            raise ValueError("task <-> data mismatch. The necessary features\
                              are not available in the supplied data")
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
        class_name = self.__class__.__name__
        estimator_name = self._estimator.__class__.__name__
        return '%s(%s(%s))' % (class_name, estimator_name,
                               _pprint(self.get_params(deep=False), offset=len(class_name), ),)


class TSCStrategy(BaseStrategy):
    """Strategies for Time Series Classification
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case = "TSC"


class TSRStrategy(BaseStrategy):
    """Strategies for Time Series Regression
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case = "TSR"


