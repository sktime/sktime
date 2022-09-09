# -*- coding: utf-8 -*-
"""
Extension template for deep learning time series classifiers.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y, classes_,
    n_classes_, fit_time_, _class_dictionary, _threads_to_use, _tags, _tags_dynamic
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details: https://www.sktime.org/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting                 - _fit(self, X, y)
    building model          - build_model(self, input_shape, n_classes, **kwargs)

Optional implements:
    data conversion and capabilities tags - _tags

Testing - implement if sktime classifier (not needed locally):
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

# todo: add any necessary imports here
# todo: import your DL Network here
from stkime.networks.mytimeseriesnetwork import MyTimeSeriesNetwork

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.utils.validation._dependencies import _check_dl_dependencies

# todo: import check_random_state
# from sktime.utils import check_random_state

# todo: if any imports are sktime soft dependencies:
#  * make sure to fill in the "python_dependencies" tag with the package import name
#  * add a _check_soft_dependencies warning here, example:
#
# from sktime.utils.validation._dependencies import check_soft_dependencies
# _check_soft_dependencies("soft_dependency_name", severity="warning")
_check_dl_dependencies(severity="warning")


class MyTimeSeriesClassifier(BaseDeepClassifier):
    """Custom time series DL classifier. todo: write docstring.

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

    Notes
    -----
    Add link of source code if you are adapting from an existing implementation.

    References
    ----------
    .. [1] Add reference of article/paper where the network was initially defined.
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
        self._network = MyTimeSeriesNetwork()

        # todo: check if dependencies are satsified
        _check_dl_dependencies(severity="error")

        # todo: change "MyTimeSeriesClassifier" to the name of the class
        super(MyTimeSeriesClassifier, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)
        n_classes: int
            The number of classes, which becomes the size of the output layer

        Returns
        -------
        output : a compiled Keras Model
        """
        # todo: import relevant dependencies
        # import tensorflow as tf
        from tensorflow import keras

        # implement here
        # todo: Set random seed for reproducibility, like:
        # tf.random.set_seed(self.random_state)
        # Note: The input and output layer defined in the networks are
        # the layers to the keras network without the output layer
        # Get the input_layer and output_layer from network like this:
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        # todo: Add a Dense layer to the output_layer
        # with number of units = n_classes like this
        # output_layer = keras.layers.Dense(units=n_classes)(output_layer)

        # todo: compile the model using the input_layer and output_layer
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile()
        return model

    # todo: implement this, mandatory
    def _fit(self, X, y):
        """Fit time series classifier to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.
        """

        # implement here
        # IMPORTANT: avoid side effects to X, y
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (X, y) or data-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit
