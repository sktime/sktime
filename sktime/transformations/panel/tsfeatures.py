# -*- coding: utf-8 -*-
"""tsfeatures interface class."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["chillerobscuro"]
__all__ = ["TSFeaturesExtractor"]

# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

from tsfeatures import tsfeatures

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("tsfeatures", severity="error")


class TSFeaturesExtractor(BaseTransformer):
    """Custom transformer. todo: write docstring.

    Parameters
    ----------
    freq : int, default = None
        Optional frequency argument passed to tsfeatures.
        If None then freq will be inferred using pd.infer_freq()
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    #
    # todo: define internal types for X, y in _fit/_transform by setting the tags
    #   X_inner_mtype - the internal mtype used for X in _fit and _transform
    #   y_inner_mtype - if y is used, the internal mtype used for y; usually "None"
    #   setting this guarantees that X, y passed to _fit, _transform are of above types
    #   for possible mtypes see datatypes.MTYPE_REGISTER, or the datatypes tutorial
    #
    #  when scitype:transform-input is set to Panel:
    #   X_inner_mtype must be changed to one or a list of sktime Panel mtypes
    #  when scitype:transform-labels is set to Series or Panel:
    #   y_inner_mtype must be changed to one or a list of compatible sktime mtypes
    #  the other tags are "safe defaults" which can usually be left as-is
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Panel",
        # valid values: "Series", "Panel", "Primitives"
        #
        # scitype:instancewise = is fit_transform an instance-wise operation?
        # instance-wise = only values of a given series instance are used to transform
        #   that instance. Example: Fourier transform; non-example: series PCA
        "scitype:instancewise": True,
        #
        # scitype:transform-labels types the y used in transform
        #   if y is not used in transform, this should be "None"
        "scitype:transform-labels": "None",
        # valid values: "None" (not needed), "Primitives", "Series", "Panel"
        #
        #
        # behavioural tags: internal type
        # ----------------------------------
        #
        # X_inner_mtype, y_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _transform, etc
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": False,  # todo - maybe true?
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, freq=None):
        self.freq = freq

        super(TSFeaturesExtractor, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # todo: add the return mtype/scitype to the docstring, e.g.,
        #  Returns
        #  -------
        #  X_transformed : Series of mtype pd.DataFrame
        #       transformed version of X

        df = X.copy()
        df = df.stack().reset_index()
        df.columns = ["ds", "unique_id", "y"]

        import pandas as pd

        if self.freq is None:
            try:
                pd.infer_freq(df.ds)
            except TypeError:
                raise TypeError(
                    "Can not infer freq from series index, "
                    "must pass explicitly to `freq` argument"
                )

        if self.freq:
            return tsfeatures(df, freq=self.freq)
        else:
            return tsfeatures(df)

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
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
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
        return [{"freq": 1}]
