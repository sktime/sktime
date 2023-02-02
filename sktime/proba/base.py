# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

__all__ = ["BaseProba"]

from sktime.base import BaseObject
from sktime.utils.validation._dependencies import _check_estimator_deps


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseProba(BaseObject):
    """Base probability distribution."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # string or str list of pkg soft dependencies
    }

    def __init__(self):
        super(BaseProba, self).__init__()
        _check_estimator_deps(self)

    def _loc(self, rowidx=None, colidx=None):
        return NotImplemented

    def _iloc(self, rowidx=None, colidx=None):
        return NotImplemented

    @property
    def loc(self):
        return _Indexer(ref=self, method="_loc")

    @property
    def iloc(self):
        return _Indexer(ref=self, method="_iloc")


class _Indexer:

    def __init__(self, ref, method="_loc"):
        self.ref = ref
        self.method = method

    def __getitem__(self, key):

        def is_noneslice(obj):
            res = isinstance(obj, slice)
            res = res and obj.start is None and obj.stop is None and obj.step is None
            return res

        ref = self.ref
        indexer = getattr(ref, self.method)

        if isinstance(key, tuple):
            if not len(key) == 2:
                raise ValueError(
                    "there should be one or two keys when calling .loc, "
                    "e.g., mydist[key], or mydist[key1, key2]"
                )
            rows = key[0]
            cols = key[1]
            if is_noneslice(rows) and is_noneslice(cols):
                return ref
            elif is_noneslice(cols):
                return indexer(rowidx=rows, colidx=None)
            elif is_noneslice(rows):
                return indexer(rowidx=None, colidx=cols)
            else:
                return indexer(rowidx=rows, colidx=cols)
        else:
            return indexer(rowidx=key[0], colidx=None)


class _BaseTFProba(BaseProba):

    def __init__(self, index=None, columns=None, distr=None):

        self.index = index
        self.columns = columns
        self.distr = distr

        super(BaseObject, self).__init__()

    def _loc(self, rowidx=None, colidx=None):
        row_iloc = self.index.get_indexer(rowidx)
        col_iloc = self.columns.get_indexer(colidx)
        return self._iloc(rowidx=row_iloc, colidx=col_iloc)

    def _iloc(self, rowidx=None, colidx=None):
        distr_subset = self.distr[rowidx, colidx]

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(subs)
            else:
                return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        return _BaseTFProba(
            index=index_subset, columns=columns_subset, distr=distr_subset,
        )
