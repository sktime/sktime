# -*- coding: utf-8 -*-
"""
rescaling.py
=========================
Signature rescaling methods. This implements the pre- and post- signature
rescaling methods along with generic scaling methods along feature
dimensions of 3D tensors.

Code for `rescale_path` and `rescale_signature` written by Patrick Kidger.
"""
import math
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    FunctionTransformer,
)
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer

from sktime.utils.check_imports import _check_soft_dependencies
_check_soft_dependencies('torch', 'signatory')  # noqa
import torch  # noqa: E402
import signatory    # noqa: E402


class TrickScaler(BaseSeriesAsFeaturesTransformer):
    """Tricks an sklearn scaler so that it uses the correct dimensions.

    This class was created out of a desire to use sklearn scaling functionality
    on 3D tensors. Sklearn operates on a tensor of shape [N, C] and
    normalises along the channel dimensions. To make this functionality work on
    tensors of shape [N, L, C] we simply first stack the first two dimensions
    to get shape [N * L, C], apply a scaling function and finally stack back
    to shape [N, L, C].

    Parameters
    ----------
    scaling : str, Scaling method, one of ['stdsc', 'maxabs', 'minmax', None].

    # TODO allow scaling to be passed as an sklearn transformer.
    """

    def __init__(self, scaling):
        self.scaling = scaling

        # Checks
        allowed_values = ["stdsc", "minmax", "maxabs", None]
        if scaling not in allowed_values:
            raise ValueError(
                "scaling param {} not recognised. Must be one "
                "of {}.".format(scaling, allowed_values)
            )

    def _trick(self, X):
        return X.reshape(-1, X.shape[2])

    def _untrick(self, X, shape):
        return X.reshape(shape)

    def fit(self, X, y=None):
        # Setup the scaler
        scaling = self.scaling
        if scaling == "stdsc":
            self.scaler = StandardScaler()
        elif scaling == "maxabs":
            self.scaler = MaxAbsScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling is None:
            self.scaler = FunctionTransformer(func=None)

        self.scaler.fit(self._trick(X), y)
        self._is_fitted = True
        return self

    def transform(self, X):
        # Input checks
        self.check_is_fitted()

        # 3d -> 2d -> 2d_scaled -> 3d_scaled
        X_tfm_2d = self.scaler.transform(self._trick(X))
        X_tfm_3d = torch.Tensor(self._untrick(X_tfm_2d, X.shape))

        return X_tfm_3d


def rescale_path(path, depth):
    """Rescales the input path by depth! ** (1 / depth), so that the last
    signature term should be roughly O(1).

    Parameters
    ----------
    path : torch.Tensor
        Input path of shape [N, L, C].
    depth : int
        Depth the signature will be computed to.

    Returns
    -------
    torch.Tensor:
        Tensor of the same shape as path, corresponding to the scaled path.
    """
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path


def rescale_signature(signature, channels, depth):
    """Rescales the output signature by multiplying the depth-d term by d!,
    with the aim that every term become ~O(1).

    Parameters
    ----------
    signature : torch.Tensor
        The signature of a path. Shape [N, Sig_dim].
    channels : int
        The number of channels of the path the signature was computed from.
    depth : int
        The depth the signature was computed to.

    Returns
    -------
    torch.Tensor:
        The signature with factorial depth scaling.
    """
    if signatory.signature_channels(channels, depth) != signature.size(-1):
        raise ValueError(
            "Given a sigtensor with {} channels, a path with {} channels and "
            "a depth of {}, which are not consistent.".format(
                signature.size(-1), channels, depth
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

    return torch.cat(terms, dim=-1)
