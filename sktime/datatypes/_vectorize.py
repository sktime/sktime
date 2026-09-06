# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Wrapper for easy vectorization/iteration of time series data.

Contains VectorizedDF class.
"""

__author__ = ["fkiraly", "hoesler"]


import itertools

import numpy as np
import pandas as pd

from sktime.datatypes._check import check_is_scitype, mtype
from sktime.datatypes._convert import convert_to
from sktime.utils.multiindex import flatten_multiindex
from sktime.utils.parallel import parallelize


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
    iter(self) or self.__iter__()
        Iterates over each Series/Panel (depending on iterate_as) in X
        as pandas.DataFrame with Index or MultiIndex (in sktime pandas format)
    len(self) or self.__len__()
        returns number of Series/Panels in X
    get_iter_indices()
        Returns pandas.(Multi)Index that are iterated over
    reconstruct(self, df_list, convert_back=False)
        Takes iterable df_list and returns as an object of is_scitype.
        Used to obtain original format after applying operations to self iterated
    """

    SERIES_SCITYPES = ["Series", "Panel", "Hierarchical"]

    def __init__(
        self,
        X,
        y=None,
        iterate_as="Series",
        is_scitype="Panel",
        iterate_cols=False,
        remember_data=True,
    ):
        if remember_data:
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

        self.is_scitype = is_scitype
        self.X_orig_mtype = X_orig_mtype

        self._check_iterate_as(iterate_as)
        self.iterate_as = iterate_as

        self._check_iterate_cols(iterate_cols)
        self.iterate_cols = iterate_cols

        self.remember_data = remember_data

        self.converter_store = dict()

        # --- Requirement 1: detect polars input ---
        try:
            import polars as pl

            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                self.is_polars = True
                self.is_lazy = isinstance(X, pl.LazyFrame)
            else:
                self.is_polars = False
                self.is_lazy = False
        except ImportError:
            self.is_polars = False
            self.is_lazy = False

        X_multiindex = self._init_conversion(X)

        if self.is_polars:
            # FIX 1: moved get_mi_cols call inside the try block so a
            # missing adapter raises a clear ImportError instead of a
            # silent pass followed by an uninformative NameError.
            try:
                from sktime.datatypes._adapter.polars import get_mi_cols
                self.X_index_cols = get_mi_cols(X_multiindex)
            except ImportError:
                raise ImportError(
                    "sktime polars adapter not found. "
                    "Ensure sktime is installed with polars support."
                )

            if self.is_lazy:
                schema = X_multiindex.collect_schema()
                all_cols = schema.names()
            else:
                all_cols = X_multiindex.columns

            self.X_mi_columns = [c for c in all_cols if c not in self.X_index_cols]
            self.X_mi_index = None
        else:
            self.X_mi_columns = X_multiindex.columns
            self.X_mi_index = X_multiindex.index

        if remember_data:
            self.X_multiindex = X_multiindex

        # FIX 3: initialise cache for lazy iteration keys so reconstruct()
        # can use the same key order as items() without a second .collect()
        self._lazy_yielded_keys = []

        self.iter_indices = self._init_iter_indices()

        self.shape = self._iter_shape()

    def _check_iterate_cols(self, iterate_cols):
        if iterate_cols not in [True, False]:
            raise ValueError(f"iterate_cols must be a boolean, found {iterate_cols}")

    def _check_iterate_as(self, iterate_as):
        if iterate_as not in self.SERIES_SCITYPES:
            raise ValueError(
                f'iterate_as must be "Series", "Panel", or "Hierarchical", '
                f"found: {iterate_as}"
            )
        if self.is_scitype == "Panel" and iterate_as == "Hierarchical":
            raise ValueError(
                'If is_scitype is "Panel", then iterate_as must be "Series" or "Panel"'
            )
        if self.is_scitype == "Series" and iterate_as != "Series":
            raise ValueError(
                'If is_scitype is "Series", then iterate_as must be "Series"'
            )

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
        """Convert X to a pandas multiindex format or keep as polars."""
        # --- Requirement 2: skip pandas coercion for polars ---
        if getattr(self, "is_polars", False):
            return X
        is_scitype = self.is_scitype
        return self._coerce_to_df(X, is_scitype, store=self.converter_store)

    def _init_iter_indices(self):
        """Initialize indices that are iterated over in vectorization."""
        iterate_as = self.iterate_as
        is_scitype = self.is_scitype
        iterate_cols = self.iterate_cols

        # --- Requirement 3: polars init_iter_indices ---
        if getattr(self, "is_polars", False):
            iter_levels = self._iter_levels(iterate_as)

            if not iter_levels:
                row_ix = None
            elif self.is_lazy:
                # Lazy path: do NOT call .collect() at init time.
                # Store the group column names for deferred iteration.
                self._lazy_group_cols = [
                    self.X_index_cols[i] for i in iter_levels
                ]
                row_ix = None
            else:
                # Eager path: use partition_by to get unique group keys
                # without calling .collect()
                group_cols = [self.X_index_cols[i] for i in iter_levels]
                X_unique = self.X_multiindex.select(group_cols).unique()
                row_ix = [
                    tuple(row.values()) if len(row) > 1
                    else list(row.values())[0]
                    for row in X_unique.iter_rows(named=True)
                ]

            if iterate_cols:
                col_ix = self.X_mi_columns
            else:
                col_ix = None
            return row_ix, col_ix

        # --- pandas path (unchanged) ---
        X_ix = self.X_mi_index

        if iterate_as == is_scitype:
            row_ix = None
        elif iterate_as == "Series":
            row_ix = X_ix.droplevel(-1).unique()
        elif iterate_as == "Panel":
            row_ix = X_ix.droplevel([-1, -2]).unique()
        else:
            raise RuntimeError(
                f"unexpected value found for attribute self.iterate_as: {iterate_as}"
                'must be "Series" or "Panel"'
            )

        if iterate_cols:
            col_ix = self.X_mi_columns
        else:
            col_ix = None

        return row_ix, col_ix

    @property
    def index(self):
        """Defaults to pandas index of X converted to pandas type."""
        return self.X_mi_index

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

    def __len__(self):
        """Return number of indices to iterate over.

        Raises
        ------
        TypeError
            If called on a lazy Polars VectorizedDF where the group count
            is unknown without triggering .collect().
        """
        # FIX 2: lazy frames cannot know their group count at init time
        # without calling .collect(). Raise a clear TypeError so callers
        # are never silently given a wrong value.
        if getattr(self, "is_lazy", False):
            raise TypeError(
                "len() is not supported for lazy Polars VectorizedDF because "
                "the number of groups cannot be determined without materializing "
                "the LazyFrame. Call .collect() on your LazyFrame first, then "
                "pass the resulting DataFrame to VectorizedDF."
            )
        return np.prod(self.shape)

    def __iter__(self):
        """Iterate over all instances.

        Returns
        -------
        A generator over all slices/instances iterated over.
        i-th element corresponds to i-th vectorization slice, rows first then cols
        Same as iterating over 2nd tuple element of self.items()
        """
        return (
            group
            for _, _, group in self.items(
                iterate_as=self.iterate_as, iterate_cols=self.iterate_cols
            )
        )

    def __getitem__(self, i: int):
        """Return the i-th element iterated over in vectorization."""
        return next(itertools.islice(self, i, None))

    def items(self, iterate_as=None, iterate_cols=None):
        """Iterate over (group name, column name, instance) tuples.

        Iteration order is "depth first" with columns being the branches and groups the
        root nodes. Groups are ordered by group keys.

        Row name is null if iterate_as is the same as scitype of data.
        Col name is null if iterate_cols is False.

        Parameters
        ----------
        iterate_as : str ("Series", "Panel"), optional, default=self.iterate_as
            scitype of the iteration
            for instance, if X is Panel and iterate_as is "Series"
                then the class will iterate over individual Series in X
            or, if X is Hierarchical and iterate_as is "Panel"
                then the class will iterate over individual Panels in X
                    (Panel = flat/non-hierarchical collection of Series)
        iterate_cols : boolean, optional, default=self.iterate_cols
            whether to iterate over columns, if true, the class will iterate over
            columns

        Returns
        -------
        A generator returning (row index, col index, instance) tuples for vectorization.
        i-th element corresponds to i-th vectorization slice, rows first then cols
        2nd tuple element is X row sub-set to 0-th tuple element, col sub-set to 1-st
        if no sub-setting takes place for row, 0-th tuple element is None
        if no sub-setting takes place for col, 1-st tuple element is None
        """
        if iterate_as is None:
            iterate_as = self.iterate_as
        self._check_iterate_as(iterate_as)

        if iterate_cols is None:
            iterate_cols = self.iterate_cols
        self._check_iterate_cols(iterate_cols)

        def _iter_cols(inst, group_name=None):
            if iterate_cols:
                if getattr(self, "is_polars", False):
                    # FIX 4: group_cols were already dropped from inst
                    # before _iter_cols is called, so X_index_cols no
                    # longer exist on inst. Select only the data column.
                    for col in self.X_mi_columns:
                        yield group_name, col, inst.select([col])
                else:
                    for col in inst.columns:
                        yield group_name, col, _enforce_index_freq(inst[[col]])
            else:
                if getattr(self, "is_polars", False):
                    yield group_name, None, inst
                else:
                    yield group_name, None, _enforce_index_freq(inst)

        iter_levels = self._iter_levels(iterate_as)

        if getattr(self, "is_polars", False):
            is_self_iter = len(iter_levels) == len(self.X_index_cols)
        else:
            is_self_iter = len(iter_levels) == self.X_mi_index.nlevels

        if is_self_iter:
            yield from _iter_cols(self.X_multiindex)
        else:
            # --- Requirement 5: polars iteration paths ---
            if getattr(self, "is_polars", False):
                group_cols = [self.X_index_cols[i] for i in iter_levels]

                if self.is_lazy:
                    # Lazy path: call .collect() exactly ONCE on the full
                    # frame, partition, re-lazify each slice, and cache
                    # keys in self._lazy_yielded_keys in the exact order
                    # slices are produced so reconstruct() never needs
                    # to re-collect independently.
                    self._lazy_yielded_keys = []
                    collected = self.X_multiindex.collect()
                    partitions = collected.partition_by(
                        group_cols, as_dict=True
                    )
                    for key, group in partitions.items():
                        # cache raw key before normalising
                        self._lazy_yielded_keys.append(key)
                        group = group.drop(group_cols).lazy()
                        if isinstance(key, tuple) and len(key) == 1:
                            key = key[0]
                        yield from _iter_cols(group, group_name=key)
                else:
                    # Eager path: use partition_by natively
                    partitions = self.X_multiindex.partition_by(
                        group_cols, as_dict=True
                    )
                    for key, group in partitions.items():
                        group = group.drop(group_cols)
                        if isinstance(key, tuple) and len(key) == 1:
                            key = key[0]
                        yield from _iter_cols(group, group_name=key)
            else:
                # --- pandas path (unchanged) ---
                if isinstance(iter_levels, (list, tuple)) and len(iter_levels) == 1:
                    # single level, groupby expects scalar
                    iter_levels = iter_levels[0]
                for name, group in self.X_multiindex.groupby(
                    level=iter_levels, sort=False
                ):
                    yield from _iter_cols(
                        group.droplevel(iter_levels), group_name=name
                    )

    def _iter_levels(self, iterate_as):
        """Get the levels to group by for iteration using iterate_as.

        Parameters
        ----------
        iterate_as: The scitype of iteration elements

        Returns
        -------
        A list of multiindex levels to group by for iteration
        """
        iter_levels = 0
        if self.is_scitype == "Panel":
            if iterate_as == "Series":
                iter_levels = 1
        if self.is_scitype == "Hierarchical":
            if iterate_as == "Panel":
                iter_levels = 2
            elif iterate_as == "Series":
                iter_levels = 1
        if getattr(self, "is_polars", False):
            return list(range(len(self.X_index_cols) - iter_levels))
        return list(range(self.X_mi_index.nlevels - iter_levels))

    def _iter_shape(self, iterate_as=None, iterate_cols=None):
        """Get the number of groups and columns to iterate over.

        Parameters
        ----------
        iterate_as: the scitype to iterate over (default self.iterate_as)
        iterate_cols: if to iterate columns (defaults to self.iterate_cols)

        Returns
        -------
        A tuple of the number of groups and columns to iterate over.
        For lazy Polars frames, the group count is -1 (unknown until collected).
        """
        if iterate_as is None:
            iterate_as = self.iterate_as

        if iterate_cols is None:
            iterate_cols = self.iterate_cols

        iter_levels = self._iter_levels(iterate_as)

        # --- Requirement 4: polars iter_shape ---
        if getattr(self, "is_polars", False):
            is_self_iter = len(iter_levels) == len(self.X_index_cols)
            if is_self_iter:
                ngroups = 1
            elif self.is_lazy:
                # FIX 2: do NOT call .collect() here. The exact group
                # count is unknowable without materializing the frame.
                # Return -1 as a sentinel. __len__ raises a clear
                # TypeError for lazy frames so callers are never given
                # a silently wrong count.
                ngroups = -1
            else:
                # Eager path: .unique().height, no .collect() needed
                group_cols = [self.X_index_cols[i] for i in iter_levels]
                ngroups = self.X_multiindex.select(group_cols).unique().height
        else:
            # --- pandas path (unchanged) ---
            is_self_iter = len(iter_levels) == self.X_mi_index.nlevels
            ngroups = (
                1
                if is_self_iter
                else self.X_multiindex.groupby(level=iter_levels).ngroups
            )

        return (
            ngroups,
            len(self.X_mi_columns) if iterate_cols else 1,
        )

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
        df_list : iterable of objects of same type and sequence as __iter__ returns.
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

        # --- Requirement 6: polars reconstruct ---
        if getattr(self, "is_polars", False):
            try:
                import polars as pl
            except ImportError:
                raise ImportError(
                    "polars is required for polars data containers"
                )

            def _get_cols(x):
                """Get column names from a polars DataFrame or LazyFrame."""
                if hasattr(x, "collect_schema"):
                    return x.collect_schema().names()
                return x.columns

            def _force_flat_polars(df_list):
                force_flat = len(df_list) > 1 and any(
                    len(_get_cols(x)) > 1 for x in df_list
                )
                all_col_idx = [
                    ix for df in df_list for ix in _get_cols(df)
                ]
                force_flat = force_flat or len(set(all_col_idx)) != len(
                    all_col_idx
                )
                return force_flat

            row_ix, col_ix = self.get_iter_indices()
            force_flat = False
            iter_levels = self._iter_levels(self.iterate_as)
            group_cols = [self.X_index_cols[i] for i in iter_levels]

            if row_ix is None and col_ix is None:
                X_mi_reconstructed = df_list[0]
            elif col_ix is None:
                if self.is_lazy:
                    # FIX 3: use self._lazy_yielded_keys which were cached
                    # by items() in the exact same order as the slices were
                    # produced. Never re-collect here — a second .collect()
                    # may return keys in a different order, attaching them
                    # to the wrong slices.
                    keys = self._lazy_yielded_keys
                else:
                    keys = row_ix

                new_df_list = []
                for idx, df in enumerate(df_list):
                    row_val = keys[idx]
                    if not isinstance(row_val, tuple):
                        row_val = (row_val,)
                    df_with_keys = df
                    for col_name, val in zip(group_cols, row_val):
                        if col_name not in _get_cols(df_with_keys):
                            df_with_keys = df_with_keys.with_columns(
                                pl.lit(val).alias(col_name)
                            )
                    cols = _get_cols(df_with_keys)
                    ordered = group_cols + [
                        c for c in cols if c not in group_cols
                    ]
                    df_with_keys = df_with_keys.select(ordered)
                    new_df_list.append(df_with_keys)
                X_mi_reconstructed = pl.concat(
                    new_df_list, how="vertical"
                )
            elif row_ix is None:
                force_flat = _force_flat_polars(df_list)
                if col_multiindex in ["flat", "multiindex"] or force_flat:
                    new_df_list = []
                    for idx, df in enumerate(df_list):
                        c_ix = col_ix[idx]
                        if isinstance(c_ix, tuple):
                            c_ix = "__".join([str(c) for c in c_ix])
                        cols = _get_cols(df)
                        rename_dict = {
                            c: f"{c_ix}__{c}"
                            for c in cols
                            if c not in group_cols
                        }
                        df = df.rename(rename_dict)
                        new_df_list.append(df)
                    X_mi_reconstructed = pl.concat(
                        new_df_list, how="horizontal"
                    )
                else:
                    X_mi_reconstructed = pl.concat(
                        df_list, how="horizontal"
                    )
            else:
                # both row_ix and col_ix are not None
                col_concats = []
                row_n = len(row_ix)
                col_n = len(col_ix)
                for i in range(row_n):
                    ith_col_block = df_list[i * col_n : (i + 1) * col_n]
                    force_flat = force_flat or _force_flat_polars(
                        ith_col_block
                    )

                    new_col_block = []
                    for idx, df in enumerate(ith_col_block):
                        if col_multiindex in ["flat", "multiindex"] or force_flat:
                            c_ix = col_ix[idx]
                            if isinstance(c_ix, tuple):
                                c_ix = "__".join([str(c) for c in c_ix])
                            cols = _get_cols(df)
                            rename_dict = {
                                c: f"{c_ix}__{c}"
                                for c in cols
                                if c not in group_cols
                            }
                            df = df.rename(rename_dict)
                        new_col_block.append(df)

                    col_concat = pl.concat(new_col_block, how="horizontal")

                    row_val = row_ix[i]
                    if not isinstance(row_val, tuple):
                        row_val = (row_val,)

                    for col_name, val in zip(group_cols, row_val):
                        if col_name not in _get_cols(col_concat):
                            col_concat = col_concat.with_columns(
                                pl.lit(val).alias(col_name)
                            )

                    cols = _get_cols(col_concat)
                    ordered = group_cols + [
                        c for c in cols if c not in group_cols
                    ]
                    col_concat = col_concat.select(ordered)
                    col_concats.append(col_concat)

                X_mi_reconstructed = pl.concat(
                    col_concats, how="vertical"
                )

            if not convert_back:
                return X_mi_reconstructed
            else:
                X_orig_mtype = self.X_orig_mtype
                is_scitype = self.is_scitype
                if X_orig_mtype is None:
                    X_orig_mtype = mtype(self.X, as_scitype=self.is_scitype)

                # --- Requirement 6: lazy convert_back returns .lazy() ---
                if self.is_lazy:
                    # Ensure we return a LazyFrame for lazy inputs
                    if not isinstance(X_mi_reconstructed, pl.LazyFrame):
                        X_mi_reconstructed = X_mi_reconstructed.lazy()
                    return X_mi_reconstructed

                X_reconstructed_orig_format = convert_to(
                    X_mi_reconstructed,
                    to_type=X_orig_mtype,
                    as_scitype=is_scitype,
                    store=self.converter_store,
                )
                return X_reconstructed_orig_format

        # === pandas path below (unchanged) ===

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
            X_mi_reconstructed = pd.DataFrame(df_list[0])
        elif col_ix is None:
            X_mi_reconstructed = pd.concat(df_list, keys=row_ix, axis=0)
        elif row_ix is None:
            force_flat = _force_flat(df_list)
            if col_multiindex in ["flat", "multiindex"] or force_flat:
                col_keys = col_ix
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
                    col_keys = col_ix
                else:
                    col_keys = None
                col_concats += [pd.concat(ith_col_block, axis=1, keys=col_keys)]

            X_mi_reconstructed = pd.concat(col_concats, keys=row_ix, axis=0)

        X_mi_index = X_mi_reconstructed.index
        X_orig_row_index = self.X_mi_index

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

    def vectorize_est(
        self,
        estimator,
        method="clone",
        args=None,
        args_rowvec=None,
        return_type="pd.DataFrame",
        rowname_default="estimators",
        colname_default="estimators",
        varname_of_self=None,
        backend=None,
        backend_params=None,
        **kwargs,
    ):
        """Vectorize application of estimator method, return results DataFrame or list.

        This function returns a `pd.DataFrame` with `estimator` fitted on
        vectorization slices of `self`. Row and column indices are the
        same as obtained from `get_iter_indices`.

        This function:

        1. takes a single `sktime` estimator or a `pd.DataFrame` with estimator entries
        2. calls `method` of estimator, with arguments as per `args`, `args_rowvec`
        3. returns the result, a list or pd.DataFrame with estimator values

        If `estimator` is a single estimator, it is broadcast to a `pd.DataFrame`.
        Elements of `args`, `args_rowvec` can be `VectorizedDF`, in which case
        they are broadcast in the application step 2.

        For a row and column of the return,
        the entry is `estimator` at the same entry (if `DataFrame`) or `estimator`,
        where `method` has been executed with the following arguments:

        * `varname=value`, where `varname`/`value` are key-value pair of `kwargs`,
          and `value` is not an instance of `VectorizedDF`, for all such `value`
        * `varname=value.loc[row,col]`,
          where `varname`/`value` are key-value pair of `kwargs` or `args`,
          and `row` and `col` are `loc` indices corresponding to row/column,
          and `value` is an instance of `VectorizedDF`, for all such `value`
        * `varname=value.loc[row]`,
          where `varname`/`value` are key-value pair of `args_rowvec`,
          and `row` and `col` are `loc` indices corresponding to row/column,
          and `value` is an instance of `VectorizedDF`, for all such `value`
        * `varname_of_self=self`, if `varname_of_self` is not `None`. Here,
          `varname_of_self` should be read as the `str` value of that variable.

        Parameters
        ----------
        estimator : one of
            a. an sktime estimator object, instance of descendant of BaseEstimator
            b. pd.DataFrame with row/col indices being `self.get_iter_indices()`,
               entries sktime estimator objects, inst. of descendants of BaseEstimator
        method : str, optional, default="clone"
            method of estimator to call with arguments in `args`, `args_rowvec`
        args : dict, optional, default=empty dict
            arguments to pass to `method` of estimator clones
            will vectorize/iterator over rows and columns
        args_rowvec : dict, optional, default=empty dict
            arguments to pass to `method` of estimator clones
            will vectorize/iterator only over rows but not over columns
        return_type : str, one of "pd.DataFrame" or "list"
            the return will be of this type;
            if `pd.DataFrame`, with row/col indices being `self.get_iter_indices()`
            if `list`, entries in sequence corresponding to `self__iter__`
        rowname_default : str, optional, default="estimators"
            used as index name of single row if no row vectorization is performed
        colname_default : str, optional, default="estimators"
            used as index name of single column if no column vectorization is performed
        varname_of_self : str, optional, default=None
            if not None, self will be passed as kwarg under name "varname_of_self"

        backend : string, by default "None".
            Parallelization backend to use for runs.
            Runs parallel evaluate if specified and ``strategy="refit"``.

            - "None": executes loop sequentially, simple list comprehension
            - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
            - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
            - "dask": uses ``dask``, requires ``dask`` package in environment
            - "dask_lazy": same as "dask", but returns delayed object instead
            - "ray": uses ``ray``, requires ``ray`` package in environment

            Parameter is passed to ``utils.parallel.parallelize``.

        backend_params : dict, optional
            additional parameters passed to the backend as config.
            Directly passed to ``utils.parallel.parallelize``.
            Valid keys depend on the value of ``backend``:

            - "None": no additional parameters, ``backend_params`` is ignored
            - "loky", "multiprocessing" and "threading":
              any valid keys for ``joblib.Parallel`` can be passed here,
              e.g., ``n_jobs``, with the exception of ``backend``
              which is directly controlled by ``backend``
            - "dask": any valid keys for ``dask.compute`` can be passed,
              e.g., ``scheduler``
            - "ray": Prevents ray from shutting down after parallelization when setting
                the "shutdown_ray" key with value "False". Takes a "logger_name" and
                a "mute_warnings" key for configuration.
                Additionally takes a "ray_remote_args" dictionary that contains valid
                keys for ray_init. E.g:
                backend_params={"shutdown_ray":False, "ray_remote_args":{"num_cpus":2}}

        kwargs : will be passed to invoked methods of estimator(s) in ``estimator``

        Returns
        -------
        pd.DataFrame, with rows and columns as the return of ``get_iter_indices``.
          If rows or columns are not vectorized over, the single index
          will be ``rowname_default`` resp ``colname_default``.
          Entries are identity references to entries of ``estimator``,
          after ``method`` executed with arguments as above.
        """
        iterate_as = self.iterate_as
        iterate_cols = self.iterate_cols

        if args is None:
            args = kwargs
        else:
            args = args.copy()
            args.update(kwargs)

        if args_rowvec is None:
            args_rowvec = {}

        if return_type not in {"pd.DataFrame", "list"}:
            raise ValueError('return_type must be one of "pd.DataFrame" or "list"')

        if varname_of_self and not isinstance(varname_of_self, str):
            raise TypeError("varname_of_self must be a string")

        def explode(d: dict, iterate_as, iterate_cols):
            if not d:
                yield from itertools.cycle([{}])

            def _to_iter(e):
                if isinstance(e, VectorizedDF):
                    it = (
                        inst
                        for _, _, inst in e.items(
                            iterate_as=iterate_as, iterate_cols=iterate_cols
                        )
                    )

                    # repeat group for each column
                    if self.iterate_cols and not iterate_cols:
                        it = itertools.chain.from_iterable(
                            itertools.repeat(el, self.shape[1]) for el in it
                        )

                    return it
                else:
                    return itertools.cycle([e])

            keys, values_with_vec = zip(*d.items())
            for values_inst in zip(*map(_to_iter, values_with_vec)):
                yield dict(zip(keys, values_inst))

        if isinstance(estimator, pd.DataFrame):
            if estimator.shape != self.shape:
                raise ValueError(
                    f"The estimator data frame must have the same shape as self. "
                    f"Expected {self.shape}, found {estimator.shape}"
                )
            estimators = (
                cell for _, row in estimator.iterrows() for cell in row.values
            )
        else:
            estimators = itertools.cycle([estimator])

        vec_zip = zip(
            self.items(),
            explode(args, iterate_as=iterate_as, iterate_cols=iterate_cols),
            explode(args_rowvec, iterate_as=iterate_as, iterate_cols=False),
            estimators,
        )

        meta = {
            "method": method,
            "varname_of_self": varname_of_self,
            "rowname_default": rowname_default,
            "colname_default": colname_default,
        }

        ret = parallelize(
            fun=self._vectorize_est_single,
            iter=vec_zip,
            meta=meta,
            backend=backend,
            backend_params=backend_params,
        )

        if return_type == "pd.DataFrame":
            df_long = pd.DataFrame(ret)
            cols_right_order = df_long.loc[:, 1].unique()
            rows_right_order = df_long.loc[:, 0].unique()

            df = df_long.pivot(index=0, columns=1, values=2)
            # DataFrame.pivot sorts the rows & columns
            # (is this a bug? see #4683 and #5108)
            # either way, we need to fix this:
            df = df.reindex(cols_right_order, axis=1)
            df = df.reindex(rows_right_order, axis=0)

            # remove "0" and "1" from index/columns name
            df.index.names = [None] * len(df.index.names)
            df.columns.name = None

            # TODO: add test case for tuple index
            try:
                df.index = pd.MultiIndex.from_tuples(df.index)
            except TypeError:
                pass
            except ValueError:
                pass
            return df
        else:  # if return_type == "list"
            return [result for _, _, result in ret]

    def _vectorize_est_single(self, vec_tuple, meta):
        """Single loop iteration of _vectorize_est_[backend]."""
        method = meta["method"]
        varname_of_self = meta["varname_of_self"]
        rowname_default = meta["rowname_default"]
        colname_default = meta["colname_default"]

        (group_name, col_name, group), args_i, args_i_rowvec, est_i = vec_tuple
        args_i.update(args_i_rowvec)

        if varname_of_self is not None:
            args_i[varname_of_self] = group

        est_i_method = getattr(est_i, method)
        est_i_result = est_i_method(**args_i)

        if group_name is None:
            group_name = rowname_default
        if col_name is None:
            col_name = colname_default

        return (group_name, col_name, est_i_result)


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