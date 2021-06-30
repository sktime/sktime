# -*- coding: utf-8 -*-
"""
Extension template for pairwise distance or kernel on tabular data.

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

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.dists_kernels import BasePairwiseTransformer

# todo: add any necessary imports here


class MyTrafoPw(BasePairwiseTransformer):
    """Custom distance/kernel. todo: write docstring.

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
    #    currently there are no tags for pairwise transformers
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

        # todo: change "MyTrafoPw" to the name of the class
        super(MyTrafoPw, self).__init__()

    # todo: implement this, mandatory
    def _transform(self, X, X2=None):
        """
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
