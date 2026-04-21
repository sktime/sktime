# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Reinstate ``gluonts.torch.modules.loss`` for unpickling Lag-Llama checkpoints.

GluonTS 0.14.x exposed loss helpers under ``gluonts.torch.modules.loss``; that
module was removed in gluonts 0.15+. HuggingFace checkpoints still reference the
old pickle global, so :func:`torch.load` fails on newer GluonTS unless this
shim is registered (see :func:`ensure_gluonts_torch_modules_loss_shim`).
"""

from __future__ import annotations

import importlib
import sys
import types

_MODULE_NAME = "gluonts.torch.modules.loss"


class DistributionLoss:
    """Loss base class comparing a Distribution (prediction) to a Tensor."""

    def __call__(self, input, target):
        raise NotImplementedError


class NegativeLogLikelihood(DistributionLoss):
    """Negative log likelihood loss, with optional variance weighting (beta)."""

    def __init__(self, beta: float = 0.0):
        self.beta = beta

    def __call__(self, input, target):
        nll = -input.log_prob(target)
        if self.beta > 0.0:
            nll = nll * (input.variance.detach() ** self.beta)
        return nll


def ensure_gluonts_torch_modules_loss_shim() -> None:
    """Load real ``gluonts.torch.modules.loss`` if present, else register a shim.

    Safe to call repeatedly. Does not replace an already-imported module.
    """
    if _MODULE_NAME in sys.modules:
        return
    try:
        importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError:
        mod = types.ModuleType(_MODULE_NAME)
        mod.__file__ = __file__
        mod.DistributionLoss = DistributionLoss
        mod.NegativeLogLikelihood = NegativeLogLikelihood
        sys.modules[_MODULE_NAME] = mod
