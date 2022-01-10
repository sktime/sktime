# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Wrapper for easy vectorization/iteration of time series data.

Contains VectorizedDF class.
"""

import pandas as pd

from sktime.datatypes._check import check_is_scitype, mtype
from sktime.datatypes._convert import convert_to


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
    iterate_as : str ("Series", "Panel")
        scitype of the iteration
        for instance, if X is Panel and iterate_as is "Series"
            then the class will iterate over individual Series in X
        or, if X is Hierarchical and iterate_as is "Panel"
            then the class will iterate over individual Panels in X
                (Panel = flat/non-hierarchical collection of Series)

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

    def __init__(self, X, y=None, iterate_as="Series", is_scitype="Panel"):

        self.X = X

        if is_scitype is None:
            possible_scitypes = ["Panel", "Hierarchical"]
            _, _, metadata = check_is_scitype(
                X, scitype=possible_scitypes, return_metadata=True
            )
            is_scitype = metadata["scitype"]
            X_orig_mtype = metadata["mtype"]
        else:
            X_orig_mtype = None

        if is_scitype is not None and is_scitype not in ["Hierarchical", "Panel"]:
            raise ValueError(
                'is_scitype must be None, "Hierarchical" or "Panel", ',
                f"found {is_scitype}",
            )
        self.iterate_as = iterate_as

        self.is_scitype = is_scitype
        self.X_orig_mtype = X_orig_mtype

        if iterate_as not in ["Series", "Panel"]:
            raise ValueError(
                f'iterate_as must be "Series" or "Panel", found {iterate_as}'
            )
        self.iterate_as = iterate_as

        if iterate_as == "Panel" and is_scitype == "Panel":
            raise ValueError(
                'If is_scitype is "Panel", then iterate_as must be "Series"'
            )

        self.converter_store = dict()

        self.X_multiindex = self._init_conversion(X)
        self.iter_indices = self._init_iter_indices()

    def _init_conversion(self, X):
        """Convert X to a pandas multiindex format."""
        is_scitype = self.is_scitype

        if is_scitype == "Panel":
            return convert_to(
                X,
                to_type="pd-multiindex",
                as_scitype="Panel",
                store=self.converter_store,
            )
        elif is_scitype == "Hierarchical":
            return convert_to(
                X,
                to_type="pd_multiindex_hier",
                as_scitype="Hierarchical",
                store=self.converter_store,
            )
        else:
            raise RuntimeError(
                f"unexpected value found for attribute self.is_scitype: {is_scitype}"
                'must be "Panel" or "Hierarchical"'
            )

    def _init_iter_indices(self):
        """Initialize indices that are iterated over in vectorization."""
        iterate_as = self.iterate_as
        X = self.X_multiindex

        if iterate_as == "Series":
            ix = X.index.droplevel(-1).unique()
            return ix
        elif iterate_as == "Panel":
            ix = X.index.droplevel([-1, -2]).unique()
            return ix
        else:
            raise RuntimeError(
                f"unexpected value found for attribute self.iterate_as: {iterate_as}"
                'must be "Series" or "Panel"'
            )

    def get_iter_indices(self):
        """Get indices that are iterated over in vectorization.

        Returns
        -------
        pandas.Index or pandas.MultiIndex
            index with unique indices that are iterated over
            use to reconstruct data frame after iteration
        """
        return self.iter_indices

    def __len__(self):
        """Return number of indices to iterate over."""
        return len(self.get_iter_indices())

    def __getitem__(self, i: int):
        """Return the i-th element iterated over in vectorization."""
        X = self.X_multiindex
        ind = self.get_iter_indices()[i]
        item = X.loc[ind]
        # pd-multiindex type (Panel case) expects these index names:
        if self.iterate_as == "Panel":
            item.index.set_names(["instances", "timepoints"], inplace=True)
        return item

    def as_list(self):
        """Shorthand to retrieve self (iterator) as list."""
        return list(self)

    def reconstruct(self, df_list, convert_back=False, overwrite_index=True):
        """Reconstruct original format from iterable of vectorization instances.

        Parameters
        ----------
        df_list : iterable of objects of same type as __getitem__ returns.
            can be self, but will in general be another object to be useful.
            Example: [some_operation(df) for df in self] that leaves types the same
        convert_back : bool, default = False
            whether to convert output back to mtype of X in __init__
            if False, the return will be a pandas.DataFrame with Index or multiIndex
            if True, the return is converted to the mtype of X in __init__
        overwrite_index : bool, default = True
            if True, the resulting return will have index overwritten by that of X
                only if applies, i.e., overwrite is possible and X had an index
            if False, no index overwrite will happen

        Returns
        -------
        X_reconstructed_orig_format : row-concatenation of df-list,
            with keys and additional level from self.get_iter_indices
            if convert_back=False, always a pd.DataFrame in a sktime MultiIndex format
                (pd-multiindex mtype for Panel, or pd_multiindex_hier for Hierarchical)
            if convert_back=True, will have same format and mtype as X input to __init__
        """
        ix = self.get_iter_indices()
        X_mi_reconstructed = pd.concat(df_list, keys=ix)

        X_mi_index = X_mi_reconstructed.index
        X_orig_index = self.X_multiindex.index
        if overwrite_index and len(X_mi_index.names) == len(X_orig_index.names):
            X_mi_reconstructed.index.set_names(X_orig_index.names)

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
