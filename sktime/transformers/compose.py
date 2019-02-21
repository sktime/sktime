"""
This is a module containing time series transformers
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from scipy import sparse


class TSColumnTransformer(ColumnTransformer):
    """Applies transformers to columns of an array or pandas DataFrame.

    Simply takes the column transformer from sklearn and adds capability to handle pandas dataframe
    """

    def __init__(
        self,
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=1,
        transformer_weights=None,
        preserve_dataframe=True,
    ):
        self.preserve_dataframe = preserve_dataframe
        super(TSColumnTransformer, self).__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
        )

    def _hstack(self, Xs):
        """
        Stacks X horizontally.
        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """
        types = set(type(X) for X in Xs)

        if self.sparse_output_:
            return sparse.hstack(Xs).tocsr()
        if self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis="columns")
        return np.hstack(Xs)


    def _validate_output(self, result):
        """
        Ensure that the output of each transformer is 2D. Otherwise
        hstack can raise an error or produce incorrect results.

        Output can also be a pd.Series which is actually a 1D
        """
        names = [name for name, _, _, _ in self._iter(fitted=True,
                                                      replace_strings=True)]
        for Xs, name in zip(result, names):
            if not (getattr(Xs, 'ndim', 0) == 2 or isinstance(Xs, pd.Series)):
                raise ValueError(
                    "The output of the '{0}' transformer should be 2D (scipy " "matrix, array, or pandas DataFrame).".format(name))
