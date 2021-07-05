# -*- coding: utf-8 -*-
"""
Extension template for clusterers.

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
- ensure interface compatibility by testing clustering/tests
- once complete: use as a local library, or contribute to sktime via PR

Mandatory implements:
    fitting            - _fit(self, X)

Optional implements:
    cluster assignment - _predict(self, X)
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.clustering.base import BaseClusterer

# todo: add any necessary imports here


class MyClusterer(BaseClusterer):
    """Custom clusterer. todo: write docstring.

    todo: describe your custom clusterer here

    Hyper-parameters
    ----------------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on

    Components
    ----------
    est : sktime.estimator, BaseEstimator descendant
        descriptive explanation of est
    est2: another estimator
        descriptive explanation of est2
    and so on
    """

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, est, parama, est2=None, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initalize below

        # todo: write any hyper-parameters and components to self
        self.est = est
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        # important: no checking or other logic should happen here

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

        # todo: change "MyClusterer" to the name of the class
        super(MyClusterer, self).__init__()

    # todo: implement this, mandatory
    def _fit(self, X):
        """
        Fit the clustering algorithm on the dataset X

            core logic

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
            panel of univariate time series to train the clustering model on

        Returns
        -------
        reference to self
        """
        # implement here
        # IMPORTANT: avoid side effects to X

    # todo: consider implementing this, optional
    # at least one of _predict and _get_fitted_params should be implemented
    def _predict(self, X):
        """
        Return cluster center index for data samples.

            core logic

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
            panel of univariate time series to cluster

        Returns
        -------
        Numpy_Array: 1D np.array of length n_instances
            Index of the cluster each sample belongs to
        """
        # implement here
        # IMPORTANT: avoid side effects to X

    # todo: consider implementing this, optional
    # this is typically important for clustering
    # at least one of _predict and _get_fitted_params should be implemented
    def _get_fitted_params(self):
        """
        Retrieves fitted parameters of cluster model

            core logic

        returns
        ----------
        param_dict: dictionary of fitted parameters
        """
        # implement here
