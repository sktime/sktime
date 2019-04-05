from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from numpy import random
from copy import deepcopy
import math

class TimeSeriesForest(ForestClassifier):
    __author__ = "Tony Bagnall"

    """Time-Series Forest Classifier.


    TimeSeriesForest: Implementation of Deng 's Time Series Forest, with minor changes
    @article
    {deng13forest,
     author = {H.Deng and G.Runger and E.Tuv and M.Vladimir},
              title = {A time series forest for classification and feature extraction},
    journal = {Information Sciences},
    volume = {239},
    year = {2013}
    
    Overview: Input n series length m
    for each tree
        sample sqrt(m) intervals
        find mean, sd and slope for each interval, concatenate to form new data set
        build decision tree on new data set
    ensemble the trees with averaged probability estimates
    
    This implementation deviates from the original in minor ways. It samples intervals with replacement and 
    does not use the splitting criteria tiny refinement described in deng13forest. This is an intentionally stripped down,
    non configurable version for use as a hive-cote component. For a configurable tree based ensemble, see
    sktime.classifiers.ensemble.TimeSeriesForestClassifier

    Parameters
    ----------
    n_trees : ensemble size, integer, optional (default = 200)
    random_seed : seed for random, integer, optional (default to no seed, I think!)
    dim_to_use     : the column of the panda passed to use, optional (default to 0)
    min_interval   : minimum width of an interval, optional (default to 3)

    Attributes
    ----------
    _num_classes    : extracted from the data
    _num_atts       : extracted from the data
    _num_intervals  : sqrt(_num_atts)
    _classifiers    : array of DecisionTree classifiers
    _min_interval   : minimum width of an interval
    _intervals      : stores indexes of all start and end points for all classifiers
    _dim_to_use     : the column of the panda passed to use (can be passed a multidimensional problem, but will only use one)
    
    """

    def __init__(self,
                n_trees=200,
                random_state=None,
                dim_to_use = 0,
                min_interval=3
                 ):
        super(TimeSeriesForest, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_trees)
        self._num_trees=n_trees
        self.random_state = random_state
        random.seed(random_state)
        self._min_interval=min_interval
        self._dim_to_use = dim_to_use
# The following set in method fit
        self._num_classes = 0
        self._num_atts = 0
        self._num_intervals = 0
        self._classifiers = []
        self._intervals=[]

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification,

        Returns
        -------
        self : object
         """
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0,self._dim_to_use],pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")
        n_samps, self._num_atts = X.shape

        self._num_classes = np.unique(y).shape[0]

        self.classes_ = list(set(y))
        self.classes_.sort()
        self._num_intervals = int(math.sqrt(self._num_atts))
        if self._num_intervals==0:
            self._num_intervals=1
        if self._num_atts <self._min_interval:
            self._min_interval=self._num_atts
        self._intervals=np.zeros((self._num_trees,3*self._num_intervals,2),dtype=int)

        for i in range(0, self._num_trees):
            transformed_x = np.empty(shape=(3*self._num_intervals,n_samps))
            for j in range(0,self._num_intervals):
                self._intervals[i][j][0]=random.randint(self._num_atts-self._min_interval)
                length=random.randint(self._num_atts-self._intervals[i][j][0]-1)
                if length < self._min_interval:
                    length=self._min_interval
                self._intervals[i][j][1]=self._intervals[i][j][0]+length
                # Transforms here, currently just hard coding it, so not configurable
                means = np.mean(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                std_dev = np.std(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                slope = self.lsq_fit(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]])
                transformed_x[3*j]=means
                transformed_x[3*j+1]=std_dev
                transformed_x[3*j+2]=slope
            tree = deepcopy(self.base_estimator)
            transformed_x=transformed_x.T
            tree.fit(transformed_x, y)
            self._classifiers.append(tree)
            return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : The training input samples.  array-like or sparse matrix of shape = [n_samps, num_atts] or a data frame.
        If a Pandas data frame is passed, the column _dim_to_use is extracted

        Returns
        -------
        output : 1D array of predictions,
        """

        proba=self.predict_proba(X)
        return [self.classes_[np.argmax(prob)] for prob in proba]

    def predict_proba(self, X):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted

        Local variables
        ----------
        n_samps     : number of cases to classify
        num_atts    : number of attributes in X, must match _num_atts determined in fit

        Returns
        -------
        output : 2D array of probabilities,
        """
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0,self._dim_to_use],pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        n_samps, num_atts = X.shape
        if num_atts != self._num_atts:
            raise TypeError(" ERROR number of attributes in the train does not match that in the test data")
        sums = np.zeros((X.shape[0],self._num_classes), dtype=np.float64)
        for i in range(0, self._num_trees):
            transformed_x = np.empty(shape=(3*self._num_intervals,n_samps),dtype=np.float32)
            for j in range(0,self._num_intervals):
                means = np.mean(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                std_dev = np.std(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                slope = self.lsq_fit(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]])
                transformed_x[3*j]=means
                transformed_x[3*j+1]=std_dev
                transformed_x[3*j+2]=slope
            transformed_x=transformed_x.T
            sums += self._classifiers[i].predict_proba(transformed_x)

        output = sums / (np.ones(self._num_classes) * self.n_estimators)
        return output

    def lsq_fit(self, Y):
        """Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        Y: series to find slope of
        """
        x = np.arange(Y.shape[1]) + 1
        slope = (np.mean(x * Y, axis=1) - np.mean(x) * np.mean(Y, axis=1)) / ((x * x).mean() - x.mean() ** 2)
        return slope

