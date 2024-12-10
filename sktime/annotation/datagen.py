"""Synthetic data generating functions."""

from sktime.detection.datagen import (
    GenBasicGauss,
    label_piecewise_normal,
    labels_with_repeats,
    piecewise_multinomial,
    piecewise_normal,
    piecewise_normal_multivariate,
    piecewise_poisson,
)

__all__ = [
    "GenBasicGauss",
    "label_piecewise_normal",
    "labels_with_repeats",
    "piecewise_multinomial",
    "piecewise_normal",
    "piecewise_normal_multivariate",
    "piecewise_poisson",
]
