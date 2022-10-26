# -*- coding: utf-8 -*-
"""Composer that creates distance from aligner."""

__author__ = ["fkiraly"]

from deprecated.sphinx import deprecated

from sktime.dists_kernels.distances.compose_from_align import (
    DistFromAligner as new_class,
)


# TODO: remove file in v0.15.0
@deprecated(
    version="0.13.4",
    reason="DistFromAligner has moved and this import will be removed in 0.15.0. Import from sktime.dists_kernels.distances",  # noqa: E501
    category=FutureWarning,
)
class DistFromAligner(new_class):
    """Distance transformer from aligner.

    Behaviour: uses aligner.get_distance on pairs to obtain distance matrix.

    Components
    ----------
    aligner: BaseAligner, must implement get_distances method
        if None, distance is equal zero
    """

    def __init__(self, aligner=None):
        super(DistFromAligner, self).__init__(aligner=aligner)
