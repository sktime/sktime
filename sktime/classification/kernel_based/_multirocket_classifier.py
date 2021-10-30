import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.base import BaseClassifier

from sklearn.utils.multiclass import class_distribution
from sklearn.pipeline import make_pipeline

from sktime.transformations.panel.rocket import MultiRocket


class MultiRocketClassifier(BaseClassifier):
    """Classifier wrapped for the MultiRocket univariate transformer using RidgeClassifierCV.

    Parameters
    ----------
    num_features             : int, number of features (default 50,000)
    max_dilations_per_kernel : int, maximum number of dilations per kernel (default 32)
    n_features_per_kernel    : int, number of features per kernel (default 4)
    normalise                : int, normalise the data
    n_jobs                   : int, optional (default=1) The number of jobs to run in
    parallel for `transform`. ``-1`` means using all processors.
    random_state             : int, random seed (optional, default None)

    Attributes
    ----------
    classifier              : MultiRocket classifier
    n_classes               : extracted from the data

    Notes
    -----
    @article{Tan2021MultiRocket,
        title={{MultiRocket}: Multiple pooling operators and transformations for fast and effective time series classification},
        author={Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and Webb, Geoffrey I},
        year={2021},
        journal={arxiv:2102.00457v3}
        }
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(
        self,
        num_features=50_000,
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        normalise=True,
        n_jobs=1,
        random_state=None,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifier = None

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(MultiRocketClassifier, self).__init__()

    def _fit(self, X, y):
        """Build a pipeline containing the MultiRocket transformer and RidgeClassifierCV.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifier = rocket_pipeline = make_pipeline(
            MultiRocket(
                num_features=self.num_features,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_features_per_kernel=self.n_features_per_kernel,
                normalise=self.normalise,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
        rocket_pipeline.fit(X, y)

        return self

    def _predict(self, X):
        """Find predictions for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        return self.classifier.predict(X)

    def _predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        dists = np.zeros((X.shape[0], self.n_classes))
        preds = self.classifier.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
