"""Transformers for index and column subsetting."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class IndexSubset(BaseTransformer):
    r"""Index subsetting transformer.

    In transform, subsets ``X`` to the indices in ``y.index``.
    If ``y`` is None, returns ``X`` without subsetting.
    numpy-based ``X`` are interpreted as having a RangeIndex starting at n,
    where n is the number of numpy rows seen so far through ``fit`` and ``update``.
    Non-pandas types are interpreted as having index as after conversion to pandas,
    via ``datatypes.convert_to``, to the ``"pd.DataFrame"`` sktime type.

    Parameters
    ----------
    index_treatment : str, optional, one of "keep" (default) or "remove"
        determines which indices are kept in ``Xt = transform(X, y)``
        "keep" = all indices in y also appear in Xt. If not present in X, NA is filled.
        "remove" = only indices that appear in both X and y are present in Xt.

    Examples
    --------
    >>> from sktime.transformations.series.subset import IndexSubset
    >>> from sktime.datasets import load_airline
    >>> X = load_airline()[0:32]
    >>> y = load_airline()[24:42]
    >>> transformer = IndexSubset()
    >>> X_subset = transformer.fit_transform(X=X, y=y)
    """

    _tags = {
        "authors": ["fkiraly"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "y_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "transform-returns-same-time-index": False,
        "fit_is_empty": False,
        "univariate-only": False,
        "capability:inverse_transform": False,
        "remember_data": True,  # remember all data seen as _X
    }

    def __init__(self, index_treatment="keep"):
        self.index_treatment = index_treatment
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Data to be transformed
        y : pd.DataFrame or pd.Series
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame or pd.Series, same type as X
            transformed version of X
        """
        if y is None:
            return X

        X = self._X

        index_treatment = self.index_treatment
        ind_X_and_y = X.index.intersection(y.index)

        if index_treatment == "remove":
            Xt = X.loc[ind_X_and_y]
        elif index_treatment == "keep":
            Xt = X.loc[ind_X_and_y]
            y_idx_frame = type(X)(index=y.index, dtype="float64")
            Xt = Xt.combine_first(y_idx_frame)
        else:
            raise ValueError(
                f'index_treatment must be one of "remove", "keep", but found'
                f' "{index_treatment}"'
            )
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"index_treatment": "remove"}
        params2 = {"index_treatment": "keep"}

        return [params1, params2]


class ColumnSelect(BaseTransformer):
    r"""Column selection transformer.

    In transform, subsets ``X`` to ``columns`` provided as hyper-parameters.

    Sequence of columns in ``Xt=transform(X)`` is as in ``columns`` hyper-parameter.
    Caveat: this means that ``transform`` may change sequence of columns,
        even if no columns are removed from ``X`` in ``transform(X)``.

    Parameters
    ----------
    columns : pandas compatible index or index coercible, optional, default = None
        columns to which X in transform is to be subset
    integer_treatment : str, optional, one of "col" (default) and "coerce"
        determines how integer index columns are treated
        "col" = subsets by column iloc index, even if columns is not in X.columns
        "coerce" = coerces to integer pandas.Index and attempts to subset
    index_treatment : str, optional, one of "remove" (default) or "keep"
        determines which column are kept in ``Xt = transform(X, y)``
        "remove" = only indices that appear in both X and columns are present in Xt.
        "keep" = all indices in columns appear in Xt. If not present in X, NA is filled.

    Examples
    --------
    >>> from sktime.transformations.series.subset import ColumnSelect
    >>> from sktime.datasets import load_longley
    >>> X = load_longley()[1]
    >>> transformer =  ColumnSelect(columns=["GNPDEFL", "POP", "FOO"])
    >>> X_subset = transformer.fit_transform(X=X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": "None",
        "transform-returns-same-time-index": True,
        "fit_is_empty": True,
        "univariate-only": False,
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,
    }

    def __init__(self, columns=None, integer_treatment="col", index_treatment="remove"):
        self.columns = columns
        self.integer_treatment = integer_treatment
        self.index_treatment = index_treatment
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : Ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.DataFrame
            transformed version of X
        """
        columns = self.columns
        integer_treatment = self.integer_treatment
        index_treatment = self.index_treatment

        if columns is None:
            return X
        if pd.api.types.is_scalar(columns):
            columns = [columns]

        columns = pd.Index(columns)

        if integer_treatment == "col" and pd.api.types.is_integer_dtype(columns):
            columns = [x for x in columns if x < len(X.columns)]
            col_idx = X.columns[columns]
            return X[col_idx]

        in_cols = columns.isin(X.columns)
        col_X_and_cols = columns[in_cols]

        if index_treatment == "remove":
            Xt = X[col_X_and_cols]
        elif index_treatment == "keep":
            Xt = X.reindex(columns=columns)
        else:
            raise ValueError(
                f'index_treatment must be one of "remove", "keep", but found'
                f' "{index_treatment}"'
            )
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"columns": None}
        params2 = {"columns": [0, 2, 3]}
        params3 = {"columns": ["a", "foo", "bar"], "index_treatment": "keep"}
        params4 = {"columns": "a", "index_treatment": "keep"}

        return [params1, params2, params3, params4]
