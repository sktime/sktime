# -*- coding: utf-8 -*-
"""
Extension template for time series classifiers.

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
- ensure interface compatibility by testing test/test_all_estimators
- once complete: use as a local library, or contribute to sktime via PR

Mandatory implements:
    fitting         - _fit(self, X, y)
    predicting      - _predict(self, X)

Optional implements:
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.classification.base import BaseClassifier

# todo: add any necessary imports here


class MyTSC(BaseClassifier):
    """Custom time series classifier. todo: write docstring.

    todo: describe your custom time series classifier here

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

    # todo: fill out estimator tags here
    _tags = {
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
    }
    # in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

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

        # todo: change "MyTSC" to the name of the class
        super(MyTSC, self).__init__()

    # todo: implement this, mandatory
    def _fit(self, X, y):
        """Fit time series classifier to training data.

        core logic

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries
        y : array-like, shape = [n_instances] - the class labels

        Returns
        -------
        self : reference to self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """

        # implement here
        # IMPORTANT: avoid side effects to X, y

    # todo: implement this, mandatory
    def _predict(self, X):
        """predicts labels for sequences in X

        core logic

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries

        Returns
        -------
        y : array-like, shape =  [n_instances] - predicted class labels
        """

        # implement here
        # IMPORTANT: avoid side effects to X

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        # implement here
