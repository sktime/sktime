class DataHolder:
    """
    Class for holdig the data, schema, resampling splits and metadata
    """
    def __init__(self, data, task, dataset_name):
        """
        Parameters
        ----------
        data: pandas DataFrame
                dataset in pandas format
        task: sktime task
            sktime task object
        dataset_name: string
            Name of the dataset
        """

        self._data=data
        self._task=task
        self._dataset_name=dataset_name
    
    @property
    def data(self):
        return self._data
    
    @property
    def task(self):
        return self._task
    
    @property
    def dataset_name(self):
        return self._dataset_name

    def set_resampling_splits(self, train_idx, test_idx):
        """
        Saves the train test indices after the data is resampled

        Parameters
        -----------
        train_idx: numpy array
            array with indices of the train set
        test_idx: numpy array
            array with indices of the test set
        """
        self._train_idx = train_idx
        self._test_idx = test_idx

class Result:
    """
    Class for storing the results of the orchestrator
    """
    def __init__(self, dataset_name, strategy_name, true_labels, predictions):
        """
        Parameters
        -----------
        dataset_name: string
            name of the dataset
        strategy_name: string
            name of strategy
        true_labels: array
            true labels 
        predictions: array
            predictions of estimator
        """

        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._true_labels = true_labels
        self._predictions = predictions

    
    @property
    def dataset_name(self):
        return self._dataset_name
    @property
    def strategy_name(self):
        return self._strategy_name
    @property
    def true_labels(self):
        return self._true_labels
    @property
    def predictions(self):
        return self._predictions
