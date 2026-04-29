# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Asserts that legacy ``series.kalman_filter`` import paths still resolve."""

import importlib
import sys
import warnings


def _reset_legacy_modules():
    """Drop cached legacy modules so the shim's warning fires again."""
    for name in list(sys.modules):
        if name.startswith("sktime.transformations.series.kalman_filter"):
            del sys.modules[name]


def test_legacy_package_import_works_and_warns():
    _reset_legacy_modules()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("sktime.transformations.series.kalman_filter")
    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "sktime.transformations.series.kalman_filter" in str(warning.message)
        for warning in w
    )


def test_legacy_public_classes_are_identical():
    from sktime.transformations.kalman_filter import (
        KalmanFilterTransformerFP,
        KalmanFilterTransformerPK,
    )
    from sktime.transformations.kalman_filter_base import BaseKalmanFilter
    from sktime.transformations.series.kalman_filter import (
        BaseKalmanFilter as B2,
        KalmanFilterTransformerFP as FP2,
        KalmanFilterTransformerPK as PK2,
        KalmanFilterTransformerSIMD as S2,
    )
    from sktime.transformations.simdkalman import KalmanFilterTransformerSIMD
    assert BaseKalmanFilter is B2
    assert KalmanFilterTransformerFP is FP2
    assert KalmanFilterTransformerPK is PK2
    assert KalmanFilterTransformerSIMD is S2
