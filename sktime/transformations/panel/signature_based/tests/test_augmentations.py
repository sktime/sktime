# -*- coding: utf-8 -*-
import numpy as np
from sktime.transformations.panel.signature_based._augmentations import (
    _AddTime,
    _BasePoint,
    _LeadLag,
    _CumulativeSum,
    _InvisibilityReset,
    _make_augmentation_pipeline,
)


def test_augmentations():
    # Build an array X, note that this is [n_sample, n_channels, length] shape.
    X = np.array([[1, 2], [1, 1], [1, 4]]).reshape(-1, 3, 2)

    # Leadlag
    leadlag = np.array(
        [[[1, 2, 1, 2], [1, 1, 1, 2], [1, 1, 1, 1], [1, 4, 1, 1], [1, 4, 1, 4]]]
    )
    assert np.allclose(_LeadLag().transform(X), leadlag)

    # Add time
    addtime = np.array([[[0.0, 1.0, 2.0], [0.5, 1.0, 1.0], [1.0, 1.0, 4.0]]])
    assert np.allclose(_AddTime().transform(X), addtime)

    # Basepoint
    basepoint = np.array([[[0.0, 0.0], [1.0, 2.0], [1.0, 1.0], [1.0, 4.0]]])
    assert np.allclose(_BasePoint().transform(X), basepoint)

    # Invisibility reset
    ir = np.array(
        [
            [
                [1.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 4.0],
                [0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    assert np.allclose(_InvisibilityReset().transform(X), ir)

    # Cumulative sum
    cumsum = np.array([[[1, 2], [2, 3], [3, 7]]])
    assert np.allclose(_CumulativeSum().transform(X), cumsum)

    # Test multiple augmentations
    pipeline = _make_augmentation_pipeline(["leadlag", "addtime"])
    lladdtime = np.array(
        [
            [
                [0.0, 1.0, 2.0, 1.0, 2.0],
                [0.25, 1.0, 1.0, 1.0, 2.0],
                [0.5, 1.0, 1.0, 1.0, 1.0],
                [0.75, 1.0, 4.0, 1.0, 1.0],
                [1.0, 1.0, 4.0, 1.0, 4.0],
            ]
        ]
    )
    assert np.allclose(pipeline.transform(X), lladdtime)
