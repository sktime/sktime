# -*- coding: utf-8 -*-
"""Signature rescaling methods.

rescaling.py
=========================
This implements the pre- and post- signature
rescaling methods along with generic scaling methods along feature
dimensions of 3D tensors.
Code for `rescale_path` and `rescale_signature` written by Patrick Kidger.
"""
import math

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("esig", severity="warning")


def _rescale_path(path, depth):
    """Rescale input path by depth! ** (1 / depth).

    Ensures last signature term should be roughly O(1).

    Parameters
    ----------
    path : np.ndarray
        Input path of shape [N, L, C].
    depth : int
        Depth the signature will be computed to.

    Returns
    -------
    np.ndarray:
        Tensor of the same shape as path, corresponding to the scaled path.
    """
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path


def _rescale_signature(signature, channels, depth):
    """Rescals the output signature by multiplying the depth-d term by d!.

    Ain is that every term become ~O(1).

    Parameters
    ----------
    signature : np.ndarray
        The signature of a path. Shape [N, Sig_dim].
    channels : int
        The number of channels of the path the signature was computed from.
    depth : int
        The depth the signature was computed to.

    Returns
    -------
    np.ndarray:
        The signature with factorial depth scaling.
    """
    import esig

    # Needed for weird esig fails
    if depth == 1:
        sigdim = channels
    elif channels == 1:
        sigdim = depth
    else:
        sigdim = esig.sigdim(channels, depth) - 1
    # Verify shape
    if sigdim != signature.shape[-1]:
        raise ValueError(
            "A path with {} channels to depth {} should yield a "
            "signature with {} features. Input signature has {} "
            "features which is inconsistent.".format(
                channels, depth, sigdim, signature.shape[-1]
            )
        )

    end = 0
    term_length = 1
    val = 1
    terms = []
    for d in range(1, depth + 1):
        start = end
        term_length *= channels
        end = start + term_length

        val *= d

        terms.append(signature[..., start:end] * val)

    return np.concatenate(terms, axis=-1)
