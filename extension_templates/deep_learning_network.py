# -*- coding: utf-8 -*-
"""
Extension template for Deep Learning time series networks.

DL Networks are classes which specify the underlying keras network used
in DL Estimators.

Purpose of this implementation template:
    quick implementation of new networks following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new network:
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
    build_network           - build_network(self, input_shape, **kwargs)

Optional implements:
    data conversion and capabilities tags - _tags

Testing - implement if sktime classifier (not needed locally):
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

# todo: add any necessary imports here

# todo: if any imports are sktime soft dependencies:
#  * make sure to fill in the "python_dependencies" tag with the package import name
#  * add a _check_soft_dependencies warning here, example:
#
# from sktime.utils.validation._dependencies import check_soft_dependencies
# _check_soft_dependencies("soft_dependency_name", severity="warning")
_check_dl_dependencies(severity="warning")


class MyTimeSeriesNetwork(BaseDeepNetwork):
    """Custom DL Network. todo: write docstring.

    todo: describe your custom time series network here

    Hyper-parameters
    ----------------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on

    Notes
    -----
    Add link of source code if you are adapting from an existing implementation.

    References
    ----------
    .. [1] Add reference of article/paper where the network was initially defined.
    """

    # these are the default values, only add if different to these.
    _tags = {
        "python_dependencies": "tensorflow"  # If you want to add more dependencies
        # add them as a list of strings
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):
        #  if estimators have default values, set None and initalize below

        _check_dl_dependencies(severity="error")
        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

        # todo: change "MyTimeSeriesNetwork" to the name of the class
        super(MyTimeSeriesNetwork, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        # implement here
        # Add an input layer taking in input of input_shape
        # and an output layer
        # The output layer will be followed by a dense layer in DL Estimator

        # todo: return the input_layer and output_layer of keras network
        # return input_layer, output_layer
