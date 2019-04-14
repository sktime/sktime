"""
Unified high-level interface for various time series related learning tasks.
"""
import pandas as pd
from sklearn.base import BaseEstimator


class _BaseTask:
    """An object that encapsulates meta-data and instructions on how to derive the target/label for the time-series
        prediction/supervised learning task.

        Parameters
        ----------
        data : pd.DataFrame
            Contains the data that the task is expected to work with.
        target : string
            The column header for the target variable to be predicted.
        features : list of string
            The column header for the target variable to be predicted.
            If omitted, every column apart from target would be a feature.
    """
    def __init__(self, metadata, target, features=None):
        self._case = None
        self._target = target
        # by default every column apart from target is a feature
        if features is None:
            self._features = metadata.columns.drop(self._target)
        else:
            # set the user-supplied feature list as read-only
            self._features = pd.Index(features)

        # glean metadata from the dataset
        self._metadata = {"nrow": metadata.shape[0],
                          "ncol": metadata.shape[1],
                          "target_type": {target: type(i) for i in metadata[self._target]},
                          "feature_type": {col: {type(i) for i in metadata[col]} for col in self._features}}

    @property
    def target(self):
        """exposes the private variable _target in a controlled way
        """
        return self._target

    @property
    def features(self):
        """exposes the private variable _features in a controlled way
        """
        return self._features

    def __getitem__(self, key):
        """provided read only access via keys to the private _meta data of the task
        """
        if key not in self._metadata.keys():
            raise KeyError
        return self._metadata[key]


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
    def __init__(self, metadata, target, features=None):
        self._case = 'TSC'
        super(TSCTask, self).__init__(metadata, target, features=features)


class ForecastingTask(_BaseTask):
    """Forecasting task.

    Parameters
    ----------
    metadata : pandas DataFrame
        Meta-data
    target : str
        Name of target variable.
    pred_horizon : list
        List of steps ahead to predict.
    features : list
        List of feature variables.
    """
    def __init__(self, metadata, target, pred_horizon, features=None):
        self._pred_horizon = pred_horizon
        super(ForecastingTask, self).__init__(metadata, target, features=features)
        self._case = 'Forecasting'

    @property
    def pred_horizon(self):
        """
        exposes the private variable _pred_horizon in a controlled way
        """
        return self._pred_horizon


class _BaseStrategy:
    """
    A meta-estimator that employs a low level estimator to
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
        self._meta = {"tags": None}

    @property
    def case(self):
        """
        exposes the private variable _case as read only
        """
        return self._case

    def __getitem__(self, key):
        """
        provided read only access via keys
        to the private _meta data
        """
        if key not in self._meta.keys():
            raise KeyError
        return self._meta[key]

    def fit(self, task, data):
        """ Fit the estimator as per task details

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
        if self._case != task.case:
            raise ValueError("Hash mismatch: the supplied data is\
                             incompatible with the task")
        # link task
        self._task = task
        # fit the estimator
        try:
            X = data[self._task.features]
            y = data[self._task.target]
        except KeyError:
            raise ValueError("task <-> data mismatch. The necessary target/features\
                              are not available in the supplied data")
        # fit the estimator
        self._estimator.fit(X, y)

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
        """calls get_params of the estimator
        """
        return self._estimator.get_params(deep=deep)

    def set_params(self, **params):
        """calls set_params of the estimator
        """
        self._estimator.set_params(**params)


class TSCStrategy(_BaseStrategy):
    """Strategies for Time Series Classification
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case = "TSC"


class TSRStrategy(_BaseStrategy):
    """Strategies for Time Series Regression
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case = "TSR"
