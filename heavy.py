'''Implementation of Task and Strategy constructs
'''


class Task:
    '''
    A data container with the task description
    '''
    def __init__(self, case, data, target, split, features=None):
        # TODO: (discuss) I added an extra score param
        '''
        Parameters
        ----------
        case : string
            The string value could be either "TSC" for time series
            classification of "TSR" for time series regression.
        data : a pandas DataFrame
        target : string
            The column header for the target variable to be predicted
        features : list of string
            The column header for the target variable to be predicted.
            If omitted, every column apart from target would be a feature.
        score : function (sklearn compatible)
            An sklearn compatible scoring function to be used for validation
        split : (iterable, iterable) or (generator, generator)
            A tuple of iterables or generators. Should not contain an
            iterator. The iterable or generator should contain row indices
            of the data Dataframe. Syntax (train, test). Train and test
            overlap check are not performed and should be taken care of
            externally.
        '''
        self._case = case
        self._data = data
        self._spec = {"target": target,
                      "features": features,
                      "train_id": split[0],
                      "test_id": split[1]}
        # by default every column apart from target is a feature
        if self._spec['features'] is None:
            self._spec['features'] = list(data.columns)
            self._spec['features'].remove(self._spec['target'])

        # glean metadata from the dataset
        self._meta = {"nrow": data.shape[0],
                      "ncol": data.shape[1],
                      # TODO: replace with something efficient
                      "target_type": {target: type(i) for i in data[target]},
                      # TODO: replace with something efficient
                      "feature_type": {col: {type(i) for i in data[col]} for \
                                       col in self._spec['features']}}

    @property
    def case(self):
        '''
        exposes the private variable _case in a controlled way
        only a getter, no setter
        '''
        return self._case

    @property
    def data(self):
        '''
        exposes the private variable _data in a controlled way
        only a getter, no setter
        '''
        return self._data

    @property
    def spec(self):
        '''
        exposes the private variable _spec in a controlled way
        only a getter, no setter
        '''
        return self._spec

    @property
    def meta(self):
        '''
        exposes the private variable _spec in a controlled way
        only a getter, no setter
        '''
        return self._meta


class Strategy:
    '''
    A meta-estimator that employs a low level estimator to
    perform a pescribed task
    '''
    def __init__(self, Estimator, **kwargs):
        # TODO: (discuss) shouldn't task be in init instead of fit?
        # TODO: (discuss) Why Estimator instead of estimator?
        '''
        Parameters
        ----------
        Estimator : The low-level estimator class (not an instance)
        **kwargs : keyword arguments for the estimator
        '''
        # construct and initialize the estimator
        self._estimator = Estimator(**kwargs)
        self._task = None
        # TODO: (discuss) why a task_type again here?
        # TODO: (discuss) why do we need a tag? use? initialization?
        self._meta = {"task_type": None,
                      "tags": None}

    @property
    def estimator(self):
        # TODO: (discus) why are we making the estimator pubic?
        # TODO: (discuss) shall we putestimator type in meta?
        '''
        exposes the private variable _estimator in a controlled way
        only a getter, no setter
        '''
        return self._estimator

    @property
    def meta(self):
        '''
        exposes the private variable _meta in a controlled way
        only a getter, no setter
        '''
        return self._meta

    def fit(self, task, data):
        # TODO: (discuss) why do we have data here?
        # TODO: (discuss) why do we have task here?
        ''' Fit the estimator as per task details

        Parameters
        ----------
        case : string
            The string value could be either "TSC" for time series
            classification of "TSR" for time series regression.
        data : a pandas DataFrame
        '''
        # set the task and update meta_data
        self._task = task
        self._meta["task_type"] = task.case
        # TODO: remove check as it will never be true
        if(task.data is None):
            # TODO: there is no setter for _data and this shouldn't be done
            self._task._data = data
        # fit the estimator
        self._estimator.fit(self._task.data.iloc[self._task.spec["train_id"],
                                                 self._task.spec["features"]],
                            self._task.data.iloc[self._task.spec["train_id"],
                                                 self._task.spec["target"]])

    def predict(self):
        '''Predict the targets for the test data

        Returns
        -------
        predictions: a pd.Dataframe or pd.Series
            returns the predictions
        '''
        predictions = self._estimator.predict(
            self._task.data.iloc[self._task.spec["test_id"],
                                 self._task.spec["features"]])
        return predictions

    def get_params(self, deep=True):
        '''calls get_params of the estimator
        '''
        self._estimator.get_params(deep=deep)

    def set_params(self, **params):
        '''calls set_params of the estimator
        '''
        self._estimator.get_params(**params)
