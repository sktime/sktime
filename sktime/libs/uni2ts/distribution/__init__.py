"""`distributions` module of `uni2ts` library."""

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    from ._base import AffineTransformed, DistributionOutput
