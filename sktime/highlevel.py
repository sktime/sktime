'''
A Helper interface for high level operations

Implements the Task and Strategy classes for
high level operations
'''
import pandas as pd


class Task:
    '''
    A data container with the task description
    '''
    def __init__(self, case, data, target, features=None):
        '''
        Parameters
        ----------
        case : string
            The string value could be either "TSC" for time series
            classification of "TSR" for time series regression.
        data : a pandas DataFrame
            Contains the data that the task is expected to work with.
        target : string
            The column header for the target variable to be predicted.
        features : list of string
            The column header for the target variable to be predicted.
            If omitted, every column apart from target would be a feature.
        '''
        self._case = case
        self._target = target
        # by default every column apart from target is a feature
        if features is None:
            self._features = data.columns.drop(self._target)
        else:
            # set the user-supplied feature list as read-only
            self._features = pd.Index(features)

        # glean metadata from the dataset
        self._meta = {"nrow": data.shape[0],
                      "ncol": data.shape[1],
                      "target_type": {target: type(i)
                                      for i in data[self._target]},
                      "feature_type": {col: {type(i) for i in data[col]}
                                       for col in self._features}}

    @property
    def case(self):
        '''
        exposes the private variable _case as read only
        '''
        return self._case

    @property
    def target(self):
        '''
        exposes the private variable _target in a controlled way
        '''
        return self._target

    @property
    def features(self):
        '''
        exposes the private variable _features in a controlled way
        '''
        return self._features

    def __getitem__(self, key):
        '''
        provided read only access via keys
        to the private _meta data of the task
        '''
        if key not in self._meta.keys():
            raise KeyError
        return self._meta[key]


class BaseStrategy:
    '''
    A meta-estimator that employs a low level estimator to
    perform a pescribed task
    '''
    def __init__(self, estimator):
        '''
        Parameters
        ----------
        estimator : An instance of an appropriately initialized
        low-level estimator
        '''
        # construct and initialize the estimator
        self._estimator = estimator
        self._case = None
        self._task = None
        self._meta = {"tags": None}

    @property
    def case(self):
        '''
        exposes the private variable _case as read only
        '''
        return self._case

    def __getitem__(self, key):
        '''
        provided read only access via keys
        to the private _meta data
        '''
        if key not in self._meta.keys():
            raise KeyError
        return self._meta[key]

    def fit(self, task, data):
        ''' Fit the estimator as per task details

        Parameters
        ----------
        task : Task
            A task initialized with the same kind of data
        data : a pandas DataFrame
            Training Data
        '''
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
        '''Predict the targets for the test data

        Parameters
        ----------
        data : a pandas DataFrame
            Prediction Data

        Returns
        -------
        predictions: a pd.Dataframe or pd.Series
            returns the predictions
        '''
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
        '''calls get_params of the estimator
        '''
        return self._estimator.get_params(deep=deep)

    def set_params(self, **params):
        '''calls set_params of the estimator
        '''
        self._estimator.set_params(**params)


class TSCStrategy(BaseStrategy):
    '''
    Strategies for Time Series Classification
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case = "TSC"


class TSRStrategy(BaseStrategy):
    '''
    Strategies for Time Series Regression
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case = "TSR"
