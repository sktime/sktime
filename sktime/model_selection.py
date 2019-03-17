'''
classes and functions for model validation
'''
from sklearn.model_selection import GridSearchCV as skGSCV
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sktime.regressors.base import BaseRegressor
from sktime.classifiers.base import BaseClassifier
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split
import numpy as np

class GridSearchCV(skGSCV):
    '''
    A wrapper to provide default scorers
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scoring is None:
            # using accuracy score as default for classifiers
            if isinstance(self.estimator, BaseClassifier):
                self.scoring = make_scorer(accuracy_score)
            # using mean squared error as default for regressors
            elif isinstance(self.estimator, BaseRegressor):
                self.scoring = make_scorer(mean_squared_error)

class SKtime_resampling:
    """
    Abstact class that all MLaut resampling strategies should inherint from
    """
    @abstractmethod
    def resample(self):
        """
        
        """




class Single_Split(SKtime_resampling):
    """
    Wrapper for sklearn.model_selection.train_test_split
    
    The constructor implements the same parameters as sklearn.model_selection.train_test_split
    
    Parameters
    ----------
    test_size : float, int or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    
    """
    def __init__(self, test_size=0.25, train_size=None, random_state=None, shuffle=True, stratify=None):
        self._test_size=test_size
        self._train_size=train_size
        self._random_state=random_state
        self._shuffle=shuffle
        self._stratify=stratify
    
    def resample(self, X,y):
        """
        Parameters
        ----------
        X : pandas DataFrame
            DataFrame with features
        y : pandas Dataframe
            DataFrame with target variables
        Returns
        -------
        train_idx, test_idx: tuple numpy arrays
            indexes of resampled dataset
        """
        idx_dts_rows = X.shape[0]
        idx_split = np.arange(idx_dts_rows)
        train_idx, test_idx =  train_test_split(idx_split, 
                                                test_size=self._test_size, 
                                                train_size=self._train_size,
                                                random_state=self._random_state,
                                                shuffle=self._shuffle,
                                                stratify=self._stratify)
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        return train_idx, test_idx

   