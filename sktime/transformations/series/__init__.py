"""Module :mod:`sktime.transformations.series` implements series transformations.

This module provides transformations that operate on individual time series,
including feature engineering, scaling and causal discovery transformations.
"""

__all__ = []

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("pgmpy>=0.1.20", severity="none"):
    from sktime.transformations.series.causal_feature_engineer import (
        CausalFeatureEngineer,
    )

    __all__.extend(["CausalFeatureEngineer"])
