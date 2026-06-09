"""Penalties and penalty functions for change and anomaly detection."""

from ._constant_penalties import (
    make_bic_penalty,
    make_chi2_penalty,
)
from ._linear_penalties import (
    make_linear_chi2_penalty,
    make_linear_penalty,
)
from ._nonlinear_penalties import make_mvcapa_penalty, make_nonlinear_chi2_penalty

CONSTANT_PENALTY_MAKERS = [
    make_bic_penalty,
    make_chi2_penalty,
]
LINEAR_PENALTY_MAKERS = [
    make_linear_penalty,
    make_linear_chi2_penalty,
]
NONLINEAR_PENALTY_MAKERS = [
    make_nonlinear_chi2_penalty,
    make_mvcapa_penalty,
]
PENALTY_MAKERS = [
    *CONSTANT_PENALTY_MAKERS,
    *LINEAR_PENALTY_MAKERS,
    *NONLINEAR_PENALTY_MAKERS,
]

__all__ = PENALTY_MAKERS
