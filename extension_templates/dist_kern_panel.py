# -*- coding: utf-8 -*-
"""
Extension template for pairwise distance or kernel between time series.

How to use this:
- this is meant as a "fill in" template for easy extension
- do NOT import this file directly - it will break
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
[- ensure interface compatibility by testing ??? no tests yet, fill this in later]
- once complete: use as a local library, or contribute to sktime via PR

Mandatory implements:
    transforming    - _transform(self, X, X2=None)

State:
    none, this is a state-free scitype

Testing:
    get default parameters for test instance(s) - get_test_params()
    create a test instance of estimator class   - create_test_instance()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.dists_kernels import BasePairwiseTransformerPanel

# todo: add any necessary imports here


class MyTrafoPwPanel(BasePairwiseTransformerPanel):
    """Custom time series distance/kernel. todo: write docstring.

    todo: describe your custom distance/kernel here

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

    # todo: fill out transformer tags here
    #  delete the tags that you *didn't* change - these defaults are inherited
    # _tags = {
    #   currently there are no tags for pairwise transformers
    # }
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

        # todo: change "MyTrafoPwPanel" to the name of the class
        super(MyTrafoPwPanel, self).__init__()

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
    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix between time series.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

            core logic

        Parameters
        ----------
        X: list of pd.DataFrame or 2D np.arrays, of length n
        X2: list of pd.DataFrame or 2D np.arrays, of length m, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        # implement here
        # IMPORTANT: avoid side effects to X, X2
        #
        # self.symmetric: bool can be inspected, True if X == X2

    # todo: return default parameters, so that a test instance can be created
    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params
