# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

__all__ = ["BaseProba"]

import numpy as np

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

    def __init__(self, index=None, columns=None):
        self.index = index
        self.columns = columns

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

    @property
    def shape(self):
        return (len(self.index), len(self.columns))


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
            return indexer(rowidx=key, colidx=None)


class _BaseTFProba(BaseProba):

    def __init__(self, index=None, columns=None, distr=None):

        self.distr = distr

        super(_BaseTFProba, self).__init__(index=index, columns=columns)

    def _loc(self, rowidx=None, colidx=None):
        if rowidx is not None:
            row_iloc = self.index.get_indexer_for(rowidx)
        else:
            row_iloc = None
        if colidx is not None:
            col_iloc = self.columns.get_indexer_for(colidx)
        else:
            col_iloc = None
        return self._iloc(rowidx=row_iloc, colidx=col_iloc)

    def _subset_params(self, rowidx, colidx):

        params = self._get_dist_params()

        subset_param_dict = {}
        for param, val in params.items():
            arr = np.array(val)
            if len(arr.shape) == 0:
                subset_param_dict
            if len(arr.shape) >= 1 and rowidx is not None:
                arr = arr[rowidx]
            if len(arr.shape) >= 2 and colidx is not None:
                arr = arr[:, colidx]
            subset_param_dict[param] = arr
        return subset_param_dict

    def _iloc(self, rowidx=None, colidx=None):
        # distr_type = type(self.distr)
        subset_params = self._subset_params(rowidx=rowidx, colidx=colidx)
        # distr_subset = distr_type(**subset_params)

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(subs)
            else:
                return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        sk_distr_type = type(self)
        return sk_distr_type(
            index=index_subset,
            columns=columns_subset,
            **subset_params,
        )

    def _get_dist_params(self):

        params = self.get_params(deep=False)
        paramnames = params.keys()
        reserved_names = ["index", "columns"]
        paramnames = [x for x in paramnames if x not in reserved_names]

        return {k: params[k] for k in paramnames}

    def to_str(self):

        params = self._get_dist_params()

        prt = f"{self.__class__.__name__}("
        for paramname, val in params.items():
            prt += f"{paramname}={val}, "
        prt = prt[:-2] + ")"

        return prt

    def __str__(self):

        return self.to_str()
