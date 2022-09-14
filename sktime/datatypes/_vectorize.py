# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Wrapper for easy vectorization/iteration of time series data.

Contains VectorizedDF class.
"""
from itertools import product

import pandas as pd

from sktime.datatypes._check import check_is_scitype, mtype
from sktime.datatypes._convert import convert_to
from sktime.utils.multiindex import flatten_multiindex


class VectorizedDF:
    """Wrapper for easy vectorization/iteration over instances.

    VectorizedDF is an iterable that returns pandas.DataFrame
        in sktime Series or Panel format.
    Elements are all Series or Panels in X, these are iterated over.

    Parameters
    ----------
    X : object in sktime compatible Panel or Hierarchical format
        the data container to vectorize over
    y : placeholder argument, not used currently
    is_scitype : str ("Panel", "Hierarchical") or None, default = "Panel"
        scitype of X, if known; if None, will be inferred
        provide to constructor if known to avoid superfluous checks
            Caution: will not conduct checks if provided, assumes checks done
    iterate_as : str ("Series", "Panel"), optional, default="Series
        scitype of the iteration
        for instance, if X is Panel and iterate_as is "Series"
            then the class will iterate over individual Series in X
        or, if X is Hierarchical and iterate_as is "Panel"
            then the class will iterate over individual Panels in X
                (Panel = flat/non-hierarchical collection of Series)
    iterate_cols : boolean, optional, default=False
        whether to iterate over columns, if true, the class will iterate over columns

    Methods
    -------
    self[i] or self.__getitem__(i)
        Returns i-th Series/Panel (depending on iterate_as) in X
        as pandas.DataFrame with Index or MultiIndex (in sktime pandas format)
    len(self) or self.__len__
        returns number of Series/Panel in X
    get_iter_indices()
        Returns pandas.(Multi)Index that are iterated over
    reconstruct(self, df_list, convert_back=False)
        Takes iterable df_list and returns as an object of is_scitype.
        Used to obtain original format after applying operations to self iterated
    """

    SERIES_SCITYPES = ["Series", "Panel", "Hierarchical"]

    def __init__(
        self, X, y=None, iterate_as="Series", is_scitype="Panel", iterate_cols=False
    ):

        self.X = X

        if is_scitype is None:
            _, _, metadata = check_is_scitype(
                X, scitype=self.SERIES_SCITYPES, return_metadata=True
            )
            is_scitype = metadata["scitype"]
            X_orig_mtype = metadata["mtype"]
        else:
            X_orig_mtype = None

        if is_scitype is not None and is_scitype not in self.SERIES_SCITYPES:
            raise ValueError(
                'is_scitype must be None, "Hierarchical", "Panel", or "Series" ',
                f"found: {is_scitype}",
            )
        self.iterate_as = iterate_as

        self.is_scitype = is_scitype
        self.X_orig_mtype = X_orig_mtype

        if iterate_as not in self.SERIES_SCITYPES:
            raise ValueError(
                f'iterate_as must be "Series", "Panel", or "Hierarchical", '
                f"found: {iterate_as}"
            )
        self.iterate_as = iterate_as

        if is_scitype == "Panel" and iterate_as == "Hierarchical":
            raise ValueError(
                'If is_scitype is "Panel", then iterate_as must be "Series" or "Panel"'
            )

        if is_scitype == "Series" and iterate_as != "Series":
            raise ValueError(
                'If is_scitype is "Series", then iterate_as must be "Series"'
            )

        if iterate_cols not in [True, False]:
            raise ValueError(f"iterate_cols must be a boolean, found {iterate_as}")
        self.iterate_cols = iterate_cols

        self.converter_store = dict()

        self.X_multiindex = self._init_conversion(X)
        self.iter_indices = self._init_iter_indices()

    def _coerce_to_df(self, obj, scitype=None, store=None, store_behaviour=None):
        """Coerce obj to a pandas multiindex format."""
        pandas_dict = {
            "Series": "pd.DataFrame",
            "Panel": "pd-multiindex",
            "Hierarchical": "pd_multiindex_hier",
            None: ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        }

        if scitype not in pandas_dict.keys():
            raise RuntimeError(
                f"unexpected value found for attribute scitype: {scitype}"
                f"must be one of {pandas_dict.keys()}"
            )

        return convert_to(
            obj,
            to_type=pandas_dict[scitype],
            as_scitype=scitype,
            store=store,
            store_behaviour=store_behaviour,
        )

    def _init_conversion(self, X):
        """Convert X to a pandas multiindex format."""
        is_scitype = self.is_scitype
        return self._coerce_to_df(X, is_scitype, store=self.converter_store)

    def _init_iter_indices(self):
        """Initialize indices that are iterated over in vectorization."""
        iterate_as = self.iterate_as
        is_scitype = self.is_scitype
        iterate_cols = self.iterate_cols
        X = self.X_multiindex

        if iterate_as == is_scitype:
            row_ix = None
        elif iterate_as == "Series":
            row_ix = X.index.droplevel(-1).unique()
        elif iterate_as == "Panel":
            row_ix = X.index.droplevel([-1, -2]).unique()
        else:
            raise RuntimeError(
                f"unexpected value found for attribute self.iterate_as: {iterate_as}"
                'must be "Series" or "Panel"'
            )

        if iterate_cols:
            col_ix = X.columns
        else:
            col_ix = None

        return row_ix, col_ix

    @property
    def index(self):
        """Defaults to pandas index of X converted to pandas type."""
        return self.X_multiindex.index

    def get_iter_indices(self):
        """Get indices that are iterated over in vectorization.

        Returns
        -------
        pair of pandas.Index or pandas.MultiIndex or None
            i-th element of list selects rows/columns in i-th iterate sub-DataFrame
            first element is row index, second element is column index
            if rows/columns are not iterated over, row/column element is None
            use to reconstruct data frame after iteration
        """
        return self.iter_indices

    def get_iloc_indexer(self, i: int):
        """Get iloc row/column indexer for i-th list element.

        Returns
        -------
        pair of int, indexes into get_iter_indices
            1st element is iloc index for 1st element of get_iter_indices
            2nd element is iloc index for 2nd element of get_iter_indices
            together, indicate iloc index of self[i] within get_iter_indices index sets
        """
        row_ix, col_ix = self.iter_indices
        if row_ix is None and col_ix is None:
            return (0, 0)
        elif row_ix is None:
            return (0, i)
        elif col_ix is None:
            return (i, 0)
        else:
            col_n = len(col_ix)
            return (i // col_n, i % col_n)

    def _iter_indices(self):
        """Get indices that are iterated over in vectorization.

        Returns
        -------
        list of pair of pandas.Index or pandas.MultiIndex
            iterable with unique indices that are iterated over
            use to reconstruct data frame after iteration
            i-th element of list selects rows/columns in i-th iterate sub-DataFrame
            first element of pair are rows, second element are columns selected
        """
        X = self.X_multiindex
        row_ix, col_ix = self.iter_indices

        if row_ix is None and col_ix is None:
            ret = [(X.index, X.columns)]
        elif row_ix is None:
            ret = product([X.index], col_ix)
        elif col_ix is None:
            ret = product(row_ix, [X.columns])
        else:  # if row_ix and col_ix are both not None
            ret = product(row_ix, col_ix)
        return list(ret)

    def __len__(self):
        """Return number of indices to iterate over."""
        row_ix, col_ix = self.iter_indices
        if row_ix is None and col_ix is None:
            return 1
        if row_ix is None:
            return len(col_ix)
        if col_ix is None:
            return len(row_ix)
        # if row_ix and col_ix are both not None
        return len(row_ix) * len(col_ix)

    def __getitem__(self, i: int):
        """Return the i-th element iterated over in vectorization."""
        X = self.X_multiindex
        row_ind, col_ind = self._iter_indices()[i]
        if isinstance(col_ind, list):
            col_ind = pd.Index(col_ind)
        elif not isinstance(col_ind, pd.Index):
            col_ind = [col_ind]
        item = X[col_ind].loc[row_ind]
        item = _enforce_index_freq(item)
        return item

    def as_list(self):
        """Shorthand to retrieve self (iterator) as list."""
        return list(self)

    def reconstruct(
        self,
        df_list,
        convert_back=False,
        overwrite_index=True,
        col_multiindex="none",
    ):
        """Reconstruct original format from iterable of vectorization instances.

        Parameters
        ----------
        df_list : iterable of objects of same type and sequence as __getitem__ returns.
            can be self, but will in general be another object to be useful.
            Example: [some_operation(df) for df in self] that leaves types the same
        convert_back : bool, optional, default = False
            whether to convert output back to mtype of X in __init__
            if False, the return will be a pandas.DataFrame with Index or multiIndex
            if True, the return is converted to the mtype of X in __init__
        overwrite_index : bool, optional, default = True
            if True, the resulting return will have index overwritten by that of X
                only if applies, i.e., overwrite is possible and X had an index
            if False, no index overwrite will happen
        col_multiindex : str, one of "none", "flat", "multiindex", default = "none"
            whether column multiindex is introduced, and if yes, what kind of,
            in case of column vectorization being applied
            "none" - df_list are simply concatenated, unless:
                If at least one var is transformed to two or more, "flat" is enforced.
                If this would cause non-unique column names, "flat" is enforced.
            "multiindex" - additional level is introduced, by names of X passed to init,
                if there would be more than one column underneath at least one level
                this is added as an additional pandas MultiIndex level
            "flat" - additional level is introduced, by names of X passed to init
                if there would be more than one column underneath at least one level
                like "multiindex", but new level is flattened to "var__colname" strings
                no new multiindex level is introduced

        Returns
        -------
        X_reconstructed_orig_format : row-concatenation of df-list,
            with keys and additional level from self.get_iter_indices
            if convert_back=False, always a pd.DataFrame in an sktime MultiIndex format
                (pd-muliindex mtype for Panel, or pd_multiindex_hier for Hierarchical)
            if convert_back=True, will have same format and mtype as X input to __init__
        """

        def coerce_to_df(x):
            if not isinstance(x, pd.DataFrame):
                return self._coerce_to_df(x)
            else:
                return x

        df_list = [coerce_to_df(x) for x in df_list]

        # condition where "flat" behaviour is enforced if "none":
        def _force_flat(df_list):
            # transformer produces at least one multivariate output
            # and there is more than one data frame to concatenate columns of
            force_flat = len(df_list) > 1 and any(len(x.columns) > 1 for x in df_list)
            # or, there would be duplicate columns, after the transformation
            all_col_idx = [ix for df in df_list for ix in df.columns]
            force_flat = force_flat or len(set(all_col_idx)) != len(all_col_idx)
            return force_flat

        row_ix, col_ix = self.get_iter_indices()
        force_flat = False
        if row_ix is None and col_ix is None:
            X_mi_reconstructed = self.X_multiindex
        elif col_ix is None:
            X_mi_reconstructed = pd.concat(df_list, keys=row_ix, axis=0)
        elif row_ix is None:
            force_flat = _force_flat(df_list)
            if col_multiindex in ["flat", "multiindex"] or force_flat:
                col_keys = self.X_multiindex.columns
            else:
                col_keys = None
            X_mi_reconstructed = pd.concat(df_list, axis=1, keys=col_keys)
        else:  # both col_ix and row_ix are not None
            col_concats = []
            row_n = len(row_ix)
            col_n = len(col_ix)
            for i in range(row_n):
                ith_col_block = df_list[i * col_n : (i + 1) * col_n]
                force_flat = force_flat or _force_flat(ith_col_block)
                if col_multiindex in ["flat", "multiindex"] or force_flat:
                    col_keys = self.X_multiindex.columns
                else:
                    col_keys = None
                col_concats += [pd.concat(ith_col_block, axis=1, keys=col_keys)]

            X_mi_reconstructed = pd.concat(col_concats, keys=row_ix, axis=0)

        X_mi_index = X_mi_reconstructed.index
        X_orig_row_index = self.X_multiindex.index

        flatten = col_multiindex == "flat" or (col_multiindex == "none" and force_flat)
        if flatten and isinstance(X_mi_reconstructed.columns, pd.MultiIndex):
            X_mi_reconstructed.columns = flatten_multiindex(X_mi_reconstructed.columns)

        if overwrite_index and len(X_mi_index.names) == len(X_orig_row_index.names):
            X_mi_reconstructed.index = X_mi_index.set_names(X_orig_row_index.names)

        if not convert_back:
            return X_mi_reconstructed
        else:
            X_orig_mtype = self.X_orig_mtype
            is_scitype = self.is_scitype
            if X_orig_mtype is None:
                X_orig_mtype = mtype(self.X, as_scitype=self.is_scitype)

            X_reconstructed_orig_format = convert_to(
                X_mi_reconstructed,
                to_type=X_orig_mtype,
                as_scitype=is_scitype,
                store=self.converter_store,
            )

            return X_reconstructed_orig_format


def _enforce_index_freq(item: pd.Series) -> pd.Series:
    """Enforce the frequency of a Series index using pd.infer_freq.

    Parameters
    ----------
    item : pd.Series
    Returns
    -------
    pd.Series
        Pandas series with the inferred frequency. If the frequency cannot be inferred
        it will stay None
    """
    if hasattr(item.index, "freq") and item.index.freq is None:
        if len(item.index) > 2:  # pandas.infer_freq errors out for length 1 or 2
            item.index.freq = pd.infer_freq(item.index)
    return item
