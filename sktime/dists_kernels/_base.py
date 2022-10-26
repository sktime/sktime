# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class templates for distances or kernels between time series, and for tabular data.

templates in this module:

    BasePairwiseTransformer - distances/kernels for tabular data
    BasePairwiseTransformerPanel - distances/kernels for time series

Interface specifications below.

---
    class name: BasePairwiseTransformer

Scitype defining methods:
    computing distance/kernel matrix (shorthand) - __call__(self, X, X2=X)
    computing distance/kernel matrix             - transform(self, X, X2=X)

Inspection methods:
    hyper-parameter inspection  - get_params()

---
    class name: BasePairwiseTransformerPanel

Scitype defining methods:
    computing distance/kernel matrix (shorthand) - __call__(self, X, X2=X)
    computing distance/kernel matrix             - transform(self, X, X2=X)

Inspection methods:
    hyper-parameter inspection  - get_params()
"""

__author__ = ["fkiraly"]

from sktime._dists_kernels_base._base import BasePairwiseTransformer as temp_base
from sktime._dists_kernels_base._base import (
    BasePairwiseTransformerPanel as temp_base_panel,
)


# todo replace with original in sktime._dists_kernels_base when distances move is
#  complete
class BasePairwiseTransformer(temp_base):
    """Base pairwise transformer for tabular or series data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

    def __init__(self):
        super(BasePairwiseTransformer, self).__init__()


# todo replace with original in sktime._dists_kernels_base when distances move is
#  complete
class BasePairwiseTransformerPanel(temp_base_panel):
    """Base pairwise transformer for panel data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

    def __init__(self):
        super(BasePairwiseTransformerPanel, self).__init__()
