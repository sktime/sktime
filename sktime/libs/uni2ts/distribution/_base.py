#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.libs.uni2ts.common.core import abstract_class_property

if _check_soft_dependencies("torch", severity="none"):
    from torch import nn
    from torch.distributions import (
        AffineTransform,
        Distribution,
        TransformedDistribution,
    )
    from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

    from sktime.libs.uni2ts.module.ts_embed import MultiOutSizeLinear

else:
    # Create Dummy class
    class nn:
        class Module:
            pass

    class Distribution:
        pass

    class TransformedDistribution:
        pass

    class MultiOutSizeLinear:
        pass


if _check_soft_dependencies("einops", severity="none"):
    from einops import rearrange

T = TypeVar("T")


# TODO: Replace with tree_map when multiple trees supported
def tree_map_multi(func: Callable, tree: [Any, "T"], *other: [Any, "T"]) -> [Any, "T"]:
    leaves, treespec = tree_flatten(tree)
    other_leaves = [tree_flatten(o)[0] for o in other]
    return_leaves = [func(*leaf) for leaf in zip(leaves, *other_leaves)]
    return tree_unflatten(return_leaves, treespec)


def convert_to_module(tree):
    if isinstance(tree, dict):
        return nn.ModuleDict(
            {key: convert_to_module(child) for key, child in tree.items()}
        )
    if isinstance(tree, (list, tuple)):
        return nn.ModuleList([convert_to_module(child) for child in tree])
    return tree


def convert_to_container(tree):
    if isinstance(tree, nn.ModuleDict):
        return {key: convert_to_container(child) for key, child in tree.items()}
    if isinstance(tree, nn.ModuleList):
        return [convert_to_container(child) for child in tree]
    return tree


class DistrParamProj(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features,
        args_dim: [int, "T"],
        domain_map,
        proj_layer,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.args_dim = args_dim
        self.domain_map = domain_map
        self.proj = convert_to_module(
            tree_map(
                lambda dim: (
                    proj_layer(in_features, dim * out_features, **kwargs)
                    if isinstance(out_features, int)
                    else proj_layer(
                        in_features,
                        tuple(dim * of for of in out_features),
                        dim=dim,
                        **kwargs,
                    )
                ),
                args_dim,
            )
        )
        self.out_size = (
            out_features if isinstance(out_features, int) else max(out_features)
        )

    def forward(self, *args):
        params_unbounded = tree_map(
            lambda proj: rearrange(
                proj(*args),
                "... (dim out_size) -> ... out_size dim",
                out_size=self.out_size,
            ),
            convert_to_container(self.proj),
        )
        params = tree_map_multi(
            lambda func, inp: func(inp), self.domain_map, params_unbounded
        )
        return params


class AffineTransformed(TransformedDistribution):
    def __init__(
        self,
        base_dist,
        loc,
        scale,
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc if loc is not None else 0.0
        self.scale = scale if scale is not None else 1.0
        super().__init__(
            base_dist,
            [AffineTransform(loc=self.loc, scale=self.scale)],
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        return self.base_dist.variance * self.scale**2


@abstract_class_property("distr_cls")
class DistributionOutput:
    distr_cls: type[Distribution] = NotImplemented

    def distribution(
        self,
        distr_params,
        loc,
        scale,
        validate_args: Optional[bool] = None,
    ):
        distr = self._distribution(distr_params, validate_args=validate_args)
        if loc is not None or scale is not None:
            distr = AffineTransformed(distr, loc=loc, scale=scale)
        return distr

    def _distribution(
        self,
        distr_params,
        validate_args: Optional[bool] = None,
    ):
        return self.distr_cls(**distr_params, validate_args=validate_args)

    @property
    @abc.abstractmethod
    def args_dim(self) -> [int, "T"]: ...

    @property
    @abc.abstractmethod
    def domain_map(self): ...

    def get_param_proj(
        self,
        in_features: int,
        out_features,
        proj_layer: Callable[..., nn.Module] = MultiOutSizeLinear,
        **kwargs: Any,
    ) -> nn.Module:
        return DistrParamProj(
            in_features=in_features,
            out_features=out_features,
            args_dim=self.args_dim,
            domain_map=self.domain_map,
            proj_layer=proj_layer,
            **kwargs,
        )
