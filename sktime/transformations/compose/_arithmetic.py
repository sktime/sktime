"""
Combining transformers compositor for sktime.

This module provides the CombineTransformers class, which applies
user-defined operations to the outputs of multiple
transformers, ensuring matching indexes and columns.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["oresthes"]

import numpy as np
import pandas as pd

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer


class CombineTransformers(_HeterogenousMetaEstimator, BaseTransformer):
    """Combination operation applied to the outputs of multiple transformers.

    Applies ``op`` to the value result of multiple transformers,
    obtaining ``op(output1, output2, ...)``, where
    ``outputi`` is the output of transformer i coerced to
    a ``numpy.ndarray``.

    If ``op`` is a numpy ufunc, the operation is applied elementwise.

    This transformer applies a user-supplied combination operation (such as addition,
    subtraction, multiplication, division, or any custom function) to the outputs of
    two or more transformers. The operation is performed across all
    outputs, and the result is returned as a single DataFrame or Series.

    All transformers must produce outputs with matching indexes and columns. If the
    indexes or columns do not match, a ValueError is raised. The operation must accept
    as many arguments as there are transformers, and should return a DataFrame or
    Series.

    Parameters
    ----------
    transformers : list of (str, transformer) tuples
        List of transformers to apply to the input data. Each tuple contains a name
        and a transformer instance. All transformers must inherit from BaseTransformer.
    op : numpy ufuncs or callable of same signature
        Function to apply to the outputs of the transformers.
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
    >>> from sktime.transformations.compose import CombineTransformers
    >>> transformers = [
    ...     ("t1", ExponentTransformer(power=2)),
    ...     ("t2", ExponentTransformer(power=1)),
    ... ]
    >>> op = CombineTransformers(transformers, op=np.divide)
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

    _steps_attr = "_transformers"
    _steps_fitted_attr = "transformers_"

    _tags = {
        # packaging info
        # --------------
        "authors": ["oresthes"],
        "maintainers": ["oresthes"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires_y": False,
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": False,
        "capability:missing_values": True,
        "capability:multivariate": True,
        "visual_block_kind": "parallel",
    }

    def __init__(self, transformers, op):
        self.transformers = transformers
        self.op = op
        self.transformers_ = self._check_estimators(
            transformers, cls_type=BaseTransformer
        )
        super().__init__()

    @property
    def _transformers(self):
        return self._get_estimator_tuples(self.transformers, clone_ests=False)

    @_transformers.setter
    def _transformers(self, value):
        self.transformers = value
        self.transformers_ = self._check_estimators(value, cls_type=BaseTransformer)

    def _fit(self, X, y=None):
        self.transformers_ = self._check_estimators(
            self.transformers, cls_type=BaseTransformer
        )
        for _, transformer in self.transformers_:
            transformer.fit(X=X, y=y)
        return self

    def _transform(self, X, y=None):
        transformers = self._get_estimator_list(self.transformers_)
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
        result = self.op(*arrays)
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
                - transformers: list of tuples
                    A list of tuples where each tuple consists of a string key and an
                    instance of a transformer.
                - operation: function
                    A NumPy universal function (ufunc) that performs an elementwise
                    operation.
        """
        from sktime.transformations.compose import Id
        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.transformations.series.func_transform import FunctionTransformer

        params1 = {
            "transformers": [
                ("t1", ExponentTransformer(power=2)),
                ("t2", ExponentTransformer(power=1)),
            ],
            "op": np.divide,
        }
        params2 = {
            "transformers": [
                ("linear", Id()),
                ("quadratic", FunctionTransformer(np.square)),
                ("exponential", ExponentTransformer()),
            ],
            "op": np.add,
        }
        return [params1, params2]
