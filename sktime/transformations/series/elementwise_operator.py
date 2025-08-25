"""
Elementwise arithmetic operator transformer for sktime.

This module provides the ElementWiseArithmeticOperator class, which applies
user-defined elementwise arithmetic operations to the outputs of multiple
transformers, ensuring matching indexes and columns.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["oresthes"]

import numpy as np
import pandas as pd

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer


class ElementWiseArithmeticOperator(BaseTransformer, _HeterogenousMetaEstimator):
    """
    Applies an elementwise arithmetic operation to the outputs of multiple transformers.

    This transformer applies a user-supplied elementwise operation (such as addition,
    subtraction, multiplication, division, or any custom function) to the outputs of
    two or more transformers. The operation is performed elementwise across all
    outputs, and the result is returned as a single DataFrame or Series.

    All transformers must produce outputs with matching indexes and columns. If the
    indexes or columns do not match, a ValueError is raised. The operation must accept
    as many arguments as there are transformers, and should return a DataFrame or
    Series.

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformers to apply to the input data. Each tuple contains a name
        and a transformer instance. All transformers must inherit from BaseTransformer.
    operation : callable
        Function to apply elementwise to the outputs of the transformers.
        Should accept N arrays/Series/DataFrames and return a DataFrame or Series.
        Examples include numpy ufuncs (e.g., np.add, np.divide) or custom functions.

    See Also
    --------
    FeatureUnion : Concatenates outputs of multiple transformers.
    sktime.transformations.series.exponent.ExponentTransformer

    References
    ----------
    Inspired by scikit-learn's FunctionTransformer and FeatureUnion.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.utils._testing.series import _make_series
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.transformations.series.elementwise_operator import (
    ...     ElementWiseArithmeticOperator,
    ... )
    >>> transformer_list = [
    ...     ("t1", ExponentTransformer(power=2)),
    ...     ("t2", ExponentTransformer(power=1)),
    ... ]
    >>> op = ElementWiseArithmeticOperator(transformer_list, operation=np.divide)
    >>> X = _make_series(n_timepoints=10, n_columns=2, random_state=42)
    >>> Xt = op.fit_transform(X)
    >>> # Xt contains the elementwise ratio of squared to original values

    Notes
    -----
    - All transformers must output DataFrames/Series with matching indexes and columns.
    - The operation must accept as many arguments as there are transformers.
    - Broadcasting is not supported; indexes and columns must match exactly.
    - This transformer is useful for combining features via arithmetic or custom logic.


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
        "visual_block_kind": "parallel",
        "authors": ["oresthes"],
        "maintainers": ["oresthes"],
    }

    def __init__(self, transformer_list, operation):
        self.transformer_list = transformer_list
        self.operation = operation
        self.transformer_list_ = self._check_estimators(
            transformer_list, cls_type=BaseTransformer
        )
        super().__init__()

    @property
    def _transformer_list(self):
        return self._get_estimator_tuples(self.transformer_list, clone_ests=False)

    @_transformer_list.setter
    def _transformer_list(self, value):
        self.transformer_list = value
        self.transformer_list_ = self._check_estimators(value, cls_type=BaseTransformer)

    def _fit(self, X, y=None):
        self.transformer_list_ = self._check_estimators(
            self.transformer_list, cls_type=BaseTransformer
        )
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
        """
        Generate a set of test parameters for the transformer.

        Parameters
        ----------
        parameter_set : str, optional
            A string that specifies which preset of parameters to use. Defaults
            to "default".

        Returns
        -------
        list of dict
            A list containing dictionaries with test parameters.
            Each dictionary includes:
                - transformer_list: list of tuples
                    A list of tuples where each tuple consists of a string key and an
                    instance of a transformer.
                - operation: function
                    A NumPy universal function (ufunc) that performs an elementwise
                    operation.
        """
        from sktime.transformations.series.exponent import ExponentTransformer

        if parameter_set == "default":
            params1 = {
                "transformer_list": [
                    ("t1", ExponentTransformer(power=2)),
                    ("t2", ExponentTransformer(power=1)),
                ],
                "operation": np.divide,
            }
            params2 = {
                "transformer_list": [
                    ("t1", ExponentTransformer(power=3)),
                    ("t2", ExponentTransformer(power=1)),
                ],
                "operation": np.add,
            }
            return [params1, params2]
        else:
            params = {
                "transformer_list": [
                    ("t1", ExponentTransformer(power=2)),
                    ("t2", ExponentTransformer(power=1)),
                ],
                "operation": np.divide,
            }
            return [params]
