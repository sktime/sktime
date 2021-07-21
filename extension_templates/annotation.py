# -*- coding: utf-8 -*-
"""
Extension template for series annotation.

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
- ensure interface compatibility by testing forecasting/tests/test_all_forecasters
        and forecasting/tests/test_sktime_forecasters
- once complete: use as a local library, or contribute to sktime via PR

Mandatory implements:
    fitting         - _fit(self, X, Y=None)
    annotating     - _predict(self, X)

Optional implements:
    updating        - _update(self, X, Y=None)

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.annotation.base import BaseSeriesAnnotator

# todo: add any necessary imports here


class MySeriesAnnotator(BaseSeriesAnnotator):
    """Custom series annotator.

    todo: write docstring, describing your custom forecaster

    Parameters
    ----------
    fmt : str {"dense", "sparse"}, optional (default="dense")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score"}, optional (default="indicator")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.

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
    def __init__(
        self,
        est,
        parama,
        est2=None,
        paramb="default",
        paramc=None,
        fmt="dense",
        labels="indicator",
    ):
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

        # todo: change "MySeriesAnnotator" to the name of the class
        super(MySeriesAnnotator, self).__init__(fmt=fmt, labels=labels)

    # todo: implement this, mandatory
    def _fit(self, X, Y=None):
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

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: implement this, mandatory
    def _predict(self, X):
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

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    def _update(self, X, Y=None):
        """Update model with new data and optional ground truth annotations.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to update model with, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        updates fitted model (attributes ending in "_")
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh
