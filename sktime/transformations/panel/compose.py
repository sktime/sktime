"""Meta Transformers module.

This module has meta-transformations that is build using the pre-existing
transformations as building blocks.
"""

__author__ = ["mloning", "sajaysurya", "fkiraly"]
__all__ = ["ColumnConcatenator"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class ColumnConcatenator(BaseTransformer):
    """Concatenate multivariate series to a long univariate series.

    Transformer that concatenates multivariate time series/panel data
    into single univariate time series/panel data by concatenating
    each individual series on top of each other from left to right.

    Uses pandas method stack() to do the concatenating

    Examples
    --------
    >>> from sktime.transformations.panel.compose import ColumnConcatenator # noqa: E501
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> concatenator = ColumnConcatenator()
    >>> concatenator.fit_transform(data)
    array([[1.],
           [4.],
           [7.],
           [2.],
           [5.],
           [8.],
           [3.],
           [6.],
           [9.]])

    Another example with panel data.

    >>> from sktime.utils._testing.panel import _make_panel
    >>> panel_data = _make_panel(n_columns = 2,
    ...                          n_instances = 2,
    ...                          n_timepoints = 3)
    >>> panel_data = concatenator.fit_transform(panel_data)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "capability:categorical_in_X": True,
    }

    def _transform(self, X, y=None):
        """Transform the data.

        Concatenate multivariate time series/panel data into long
        univariate time series/panel
        data by simply concatenating times series in time.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and single
          column
        """
        Xst = pd.DataFrame(X.stack())
        Xt = Xst.swaplevel(-2, -1).sort_index().droplevel(-2)

        # the above has the right structure, but the wrong index
        # the time index is in general non-unique now, we replace it by integer index
        levels = list(range(Xt.index.nlevels - 1))
        inst_arr = [Xt.index.get_level_values(level) for level in levels]
        inst_idx = pd.MultiIndex.from_arrays(inst_arr)

        t_idx = [range(len(Xt.loc[x])) for x in inst_idx.unique()]
        t_idx = np.concatenate(t_idx)

        Xt.index = pd.MultiIndex.from_arrays(inst_arr + [t_idx])
        Xt.index.names = X.index.names
        return Xt
