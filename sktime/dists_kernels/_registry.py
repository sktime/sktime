# -*- coding: utf-8 -*-
"""Registry for distance kernels."""
__author__ = [
    "achieveordie",
]

from typing import NamedTuple, Set, Union

from sktime.dists_kernels._base import (
    BasePairwiseTransformer,
    BasePairwiseTransformerPanel,
)
from sktime.dists_kernels.compose_tab_to_panel import AggrDist, FlatDist
from sktime.dists_kernels.dtw import DtwDist
from sktime.dists_kernels.dummy import ConstantPwTrafoPanel
from sktime.dists_kernels.edit_dist import EditDist
from sktime.dists_kernels.scipy_dist import ScipyDist
from sktime.dists_kernels.signature_kernel import SignatureKernel


class DistKernelMetricInfo(NamedTuple):
    """Define a registry entry for a distance kernel."""

    # Name of the distance kernel
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # An instance of the Distance Kernel
    dist_kernel_instance: Union[BasePairwiseTransformer, BasePairwiseTransformerPanel]


# Registry to store all distance kernels, newer kernels are to be added here to be
# registered as a valid sktime distance kernel.
_VALID_DIST_KERNELS = [
    DistKernelMetricInfo(
        canonical_name="AggrDist",
        aka={"AggrDist", "aggr", "distance_aggregation", "aggregation_distance"},
        dist_kernel_instance=AggrDist,
    ),
    DistKernelMetricInfo(
        canonical_name="DtwDist",
        aka={"DtwDist", "dtw", "dynamic_time_warping"},
        dist_kernel_instance=DtwDist,
    ),
    DistKernelMetricInfo(
        canonical_name="EditDist",
        aka={"EditDist", "edit_distance", "edit", "edit_dist"},
        dist_kernel_instance=EditDist,
    ),
    DistKernelMetricInfo(
        canonical_name="FlatDist",
        aka={"FlatDist", "flat_distance", "flat", "flat_dist"},
        dist_kernel_instance=FlatDist,
    ),
    DistKernelMetricInfo(
        canonical_name="ScipyDist",
        aka={"ScipyDist", "scipy_distance", "scipy", "scipy_dist"},
        dist_kernel_instance=ScipyDist,
    ),
    DistKernelMetricInfo(
        canonical_name="ConstantPwTrafoPanel",
        aka={"ConstantPwtrafoPanel", "constant_pairwise", "constant"},
        dist_kernel_instance=ConstantPwTrafoPanel,
    ),
    DistKernelMetricInfo(
        canonical_name="SignatureKernel",
        aka={"SignatureKernel", "signature", "sig"},
        dist_kernel_instance=SignatureKernel,
    ),
]
