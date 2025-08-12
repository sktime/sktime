# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extension template for transformers, SIMPLE version.

Contains only bare minimum of implementation requirements for a functional transformer.
Also assumes *no composition*, i.e., no transformer or other estimator components.
For advanced cases (inverse transform, composition, etc),
    see full extension template in transformer.py

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y,
    _converter_store_X, transformers_, _tags, _tags_dynamic
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory methods to implement:
    fitting         - _fit(self, X, y=None)
    transformation  - _transform(self, X, y=None)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
__author__ = ["oresthes"]

# todo: add any necessary sktime external imports here

from sktime.transformations.base import BaseTransformer
from sktime.base import _HeterogenousMetaEstimator

import pandas as pd
import numpy as np


class ElementWiseArithmeticOperator(BaseTransformer, _HeterogenousMetaEstimator):
    """Element-wise arithmetic operator for transformer outputs.

    Applies a user-supplied elementwise operation to the outputs of two or more transformers.

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformers to apply to the input data.
    operation : callable
        Function to apply elementwise to the outputs of the transformers.
        Should accept N arrays/Series/DataFrames and return a DataFrame or Series.
    """

    _steps_attr = "_transformer_list"
    _steps_fitted_attr = "transformer_list_"

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "authors": ["oresthes"],
        "maintainers": ["oresthes"],
    }

    def __init__(self, transformer_list, operation):
        self.transformer_list = transformer_list
        self.operation = operation
        self.transformer_list_ = self._check_estimators(transformer_list, cls_type=BaseTransformer)
        super().__init__()

    @property
    def _transformer_list(self):
        return self._get_estimator_tuples(self.transformer_list, clone_ests=False)

    @_transformer_list.setter
    def _transformer_list(self, value):
        self.transformer_list = value
        self.transformer_list_ = self._check_estimators(value, cls_type=BaseTransformer)

    def _fit(self, X, y=None):
        self.transformer_list_ = self._check_estimators(self.transformer_list, cls_type=BaseTransformer)
        for _, transformer in self.transformer_list_:
            transformer.fit(X=X, y=y)
        return self

    def _transform(self, X, y=None):
        transformers = self._get_estimator_list(self.transformer_list_)
        Xt_list = [trafo.transform(X, y) for trafo in transformers]

        # Ensure all outputs have matching indexes and columns
        first_index = Xt_list[0].index
        first_columns = Xt_list[0].columns
        for i, Xt in enumerate(Xt_list[1:], 1):
            if not Xt.index.equals(first_index):
                raise ValueError(f"Index mismatch between transformer 0 and {i}")
            if not Xt.columns.equals(first_columns):
                raise ValueError(f"Column mismatch between transformer 0 and {i}")

        # Apply the operation elementwise
        # Stack into 3D array if possible, else apply operation column-wise
        arrays = [Xt.values for Xt in Xt_list]
        result = self.operation(*arrays)
        # If result is ndarray, wrap as DataFrame
        if isinstance(result, np.ndarray):
            result = pd.DataFrame(result, index=first_index, columns=first_columns)
        elif isinstance(result, pd.Series):
            result = result.to_frame()
        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sktime.transformations.series.exponent import ExponentTransformer
        # Example: ratio of two transformers
        params = {
            "transformer_list": [
                ("t1", ExponentTransformer(power=2)),
                ("t2", ExponentTransformer(power=1)),
            ],
            "operation": np.divide,
        }
        return [params]
