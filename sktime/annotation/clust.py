"""Extension template for series annotation.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, X, Y=None)
    annotating     - _predict(self, X)

Optional implements:
    updating        - _update(self, X, Y=None)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

import pandas as pd
from sklearn.cluster import KMeans

from sktime.annotation.base import BaseSeriesAnnotator


class ClusterSegmenter(BaseSeriesAnnotator):
    """Cluster-based Time Series Segmentation.

    time series segmentation using clustering is simple task. This annotator
    segments time series data into distinct segments based on similarity, identified
    using the choosen clustering algorithm.

    Parameters
    ----------
    clusterer : sklearn.cluster
        The instance of clustering algorithm used for segmentation.
    n_clusters : int, default=3
        The number of clusters to form

    """

    _tags = {
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, clusterer=None, n_clusters=3):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.clusterer = (
            clusterer if clusterer is not None else KMeans(n_clusters=n_clusters)
        )
        self.n_clusters = n_clusters

        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
    def fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        self.n_instances, self.n_timepoints = X.shape
        X_flat = X.values.reshape(-1, 1)
        self.clusterer.fit(X_flat)
        return self

    # todo: implement this, mandatory
    def predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        X_flat = X.values.reshape(-1, 1)
        labels = self.clusterer.predict(X_flat)
        labels = labels.reshape(self.n_instances, self.n_timepoints)
        return pd.DataFrame(labels, index=X.index)

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}

        """
        return {"clusterer": KMeans(n_clusters=2), "n_clusters": 2}
