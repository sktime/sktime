# -*- coding: utf-8 -*-
"""Test the signature module augmentations."""
import numpy as np

from sktime.transformations.panel.signature_based._augmentations import (
    _AddTime,
    _BasePoint,
    _CumulativeSum,
    _InvisibilityReset,
    _LeadLag,
    _make_augmentation_pipeline,
)


def test_augmentations():
    """Test the signature module augmentations."""
    # Build an array X, note that this is [n_sample, n_channels, length] shape.
    X = np.array([[1, 2], [1, 1], [1, 4]]).reshape(-1, 3, 2)
    X = np.swapaxes(X, 1, 2)

    # Leadlag
    leadlag = np.array(
        [[[1, 2, 1, 2], [1, 1, 1, 2], [1, 1, 1, 1], [1, 4, 1, 1], [1, 4, 1, 4]]]
    )
    leadlag = np.swapaxes(leadlag, 1, 2)
    assert np.allclose(_LeadLag().fit_transform(X), leadlag)

    # Add time
    addtime = np.array([[[0.0, 1.0, 2.0], [0.5, 1.0, 1.0], [1.0, 1.0, 4.0]]])
    addtime = np.swapaxes(addtime, 1, 2)
    assert np.allclose(_AddTime().fit_transform(X), addtime)

    # Basepoint
    basepoint = np.array([[[0.0, 0.0], [1.0, 2.0], [1.0, 1.0], [1.0, 4.0]]])
    basepoint = np.swapaxes(basepoint, 1, 2)
    assert np.allclose(_BasePoint().fit_transform(X), basepoint)

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
    ir = np.swapaxes(ir, 1, 2)
    assert np.allclose(_InvisibilityReset().fit_transform(X), ir)

    # Cumulative sum
    cumsum = np.array([[[1, 2], [2, 3], [3, 7]]])
    cumsum = np.swapaxes(cumsum, 1, 2)
    assert np.allclose(_CumulativeSum().fit_transform(X), cumsum)

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
    lladdtime = np.swapaxes(lladdtime, 1, 2)
    assert np.allclose(pipeline.fit_transform(X), lladdtime)
