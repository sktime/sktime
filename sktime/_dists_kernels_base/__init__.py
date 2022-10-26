# -*- coding: utf-8 -*-
"""Module exports for dist_kernels module."""

from sktime._dists_kernels_base._base import (
    BasePairwiseTransformer,
    BasePairwiseTransformerPanel,
)

# TODO: remove dir in 0.15.0
__all__ = [
    "BasePairwiseTransformer",
    "BasePairwiseTransformerPanel",
]
