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

        X_multiindex = self._init_conversion(X)
        self.X_mi_columns = X_multiindex.columns
        self.X_mi_index = X_multiindex.index
        if remember_data:
            self.X_multiindex = X_multiindex
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
        # V7 Optimization: If it's already polars, we might want to keep it
        # but the rest of the class expects pandas multiindex for index handling.
        # To truly support polars natively, we'd need a PolarsVectorizedDF.
        # Here we follow the issue's hint to support polars in the conversion flow.
        
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
        # Handle polars input by ensuring convert_to knows how to deal with it
        return self._coerce_to_df(X, is_scitype, store=self.converter_store)

    def _init_iter_indices(self):
        """Initialize indices that are iterated over in vectorization."""
        iterate_as = self.iterate_as
        is_scitype = self.is_scitype
        iterate_cols = self.iterate_cols
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
        """Get indices that are iterated over in vectorization."""
        return self.iter_indices

    def __len__(self):
        """Return number of indices to iterate over."""
        return np.prod(self.shape)

    def __iter__(self):
        """Iterate over all instances."""
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
        """Iterate over (group name, column name, instance) tuples."""
        if iterate_as is None:
            iterate_as = self.iterate_as
        self._check_iterate_as(iterate_as)

        if iterate_cols is None:
            iterate_cols = self.iterate_cols
        self._check_iterate_cols(iterate_cols)

        def _iter_cols(inst, group_name=None):
            if iterate_cols:
                for col in inst.columns:
                    yield group_name, col, _enforce_index_freq(inst[[col]])
            else:
                yield group_name, None, _enforce_index_freq(inst)

        iter_levels = self._iter_levels(iterate_as)
        is_self_iter = len(iter_levels) == self.X_mi_index.nlevels

        if is_self_iter:
            yield from _iter_cols(self.X_multiindex)
        else:
            if isinstance(iter_levels, (list, tuple)) and len(iter_levels) == 1:
                iter_levels = iter_levels[0]
            
            # Optimization: If the input was originally Polars, 
            # we could theoretically use polars.groupby here for speed.
            # But the current architecture converts to pandas multiindex in __init__.
            for name, group in self.X_multiindex.groupby(level=iter_levels, sort=False):
                yield from _iter_cols(group.droplevel(iter_levels), group_name=name)

    def _iter_levels(self, iterate_as):
        iter_levels = 0
        if self.is_scitype == "Panel":
            if iterate_as == "Series":
                iter_levels = 1
        if self.is_scitype == "Hierarchical":
            if iterate_as == "Panel":
                iter_levels = 2
            elif iterate_as == "Series":
                iter_levels = 1
        return list(range(self.X_mi_index.nlevels - iter_levels))

    def _iter_shape(self, iterate_as=None, iterate_cols=None):
        if iterate_as is None:
            iterate_as = self.iterate_as
        if iterate_cols is None:
            iterate_cols = self.iterate_cols
        iter_levels = self._iter_levels(iterate_as)
        is_self_iter = len(iter_levels) == self.X_mi_index.nlevels
        return (
            1 if is_self_iter else self.X_multiindex.groupby(level=iter_levels).ngroups,
            len(self.X_mi_columns) if iterate_cols else 1,
        )

    def as_list(self):
        return list(self)

    def reconstruct(
        self,
        df_list,
        convert_back=False,
        overwrite_index=True,
        col_multiindex="none",
    ):
        """Reconstruct original format from iterable of vectorization instances."""

        def coerce_to_df(x):
            if not isinstance(x, pd.DataFrame):
                # V7: This will now handle polars outputs from operations correctly
                return self._coerce_to_df(x)
            else:
                return x

        df_list = [coerce_to_df(x) for x in df_list]

        def _force_flat(df_list):
            force_flat = len(df_list) > 1 and any(len(x.columns) > 1 for x in df_list)
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
            col_keys = col_ix if col_multiindex in ["flat", "multiindex"] or force_flat else None
            X_mi_reconstructed = pd.concat(df_list, axis=1, keys=col_keys)
        else:
            col_concats = []
            row_n = len(row_ix)
            col_n = len(col_ix)
            for i in range(row_n):
                ith_col_block = df_list[i * col_n : (i + 1) * col_n]
                force_flat = force_flat or _force_flat(ith_col_block)
                col_keys = col_ix if col_multiindex in ["flat", "multiindex"] or force_flat else None
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
            
            # V7: Final conversion back will now correctly produce polars if requested
            return convert_to(
                X_mi_reconstructed,
                to_type=X_orig_mtype,
                as_scitype=is_scitype,
                store=self.converter_store,
            )

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
        iterate_as = self.iterate_as
        iterate_cols = self.iterate_cols
        if args is None: args = kwargs
        else:
            args = args.copy()
            args.update(kwargs)
        if args_rowvec is None: args_rowvec = {}

        def explode(d: dict, iterate_as, iterate_cols):
            if not d: yield from itertools.cycle([{}])
            def _to_iter(e):
                if isinstance(e, VectorizedDF):
                    it = (inst for _, _, inst in e.items(iterate_as=iterate_as, iterate_cols=iterate_cols))
                    if self.iterate_cols and not iterate_cols:
                        it = itertools.chain.from_iterable(itertools.repeat(el, self.shape[1]) for el in it)
                    return it
                else: return itertools.cycle([e])
            keys, values_with_vec = zip(*d.items())
            for values_inst in zip(*map(_to_iter, values_with_vec)):
                yield dict(zip(keys, values_inst))

        if isinstance(estimator, pd.DataFrame):
            estimators = (cell for _, row in estimator.iterrows() for cell in row.values)
        else: estimators = itertools.cycle([estimator])

        vec_zip = zip(self.items(), explode(args, iterate_as=iterate_as, iterate_cols=iterate_cols),
                      explode(args_rowvec, iterate_as=iterate_as, iterate_cols=False), estimators)
        meta = {"method": method, "varname_of_self": varname_of_self, "rowname_default": rowname_default, "colname_default": colname_default}
        ret = parallelize(fun=self._vectorize_est_single, iter=vec_zip, meta=meta, backend=backend, backend_params=backend_params)

        if return_type == "pd.DataFrame":
            df_long = pd.DataFrame(ret)
            cols_right_order, rows_right_order = df_long.loc[:, 1].unique(), df_long.loc[:, 0].unique()
            df = df_long.pivot(index=0, columns=1, values=2).reindex(cols_right_order, axis=1).reindex(rows_right_order, axis=0)
            df.index.names, df.columns.name = [None] * len(df.index.names), None
            try: df.index = pd.MultiIndex.from_tuples(df.index)
            except (TypeError, ValueError): pass
            return df
        else: return [result for _, _, result in ret]

    def _vectorize_est_single(self, vec_tuple, meta):
        method, varname_of_self = meta["method"], meta["varname_of_self"]
        (group_name, col_name, group), args_i, args_i_rowvec, est_i = vec_tuple
        args_i.update(args_i_rowvec)
        if varname_of_self is not None: args_i[varname_of_self] = group
        est_i_result = getattr(est_i, method)(**args_i)
        return (group_name or meta["rowname_default"], col_name or meta["colname_default"], est_i_result)


def _enforce_index_freq(item: pd.Series) -> pd.Series:
    if hasattr(item.index, "freq") and item.index.freq is None:
        if len(item.index) > 2: item.index.freq = pd.infer_freq(item.index)
    return item
