# -*- coding: utf-8 -*-
"""
compute.py
=======================
Class for signature computation over windows.
"""
import numpy as np
from sktime.transformations.base import _SeriesToPrimitivesTransformer
from sktime.transformations.panel.signature_based._window import _window_getter
from sktime.transformations.panel.signature_based._rescaling import (
    _rescale_path,
    _rescale_signature,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("esig")
import esig  # noqa: E402


class _WindowSignatureTransform(_SeriesToPrimitivesTransformer):
    """Performs the signature transform over given windows.
    Given data of shape [N, L, C] and specification of a window method from the
    signatures window module, this class will compute the signatures over
    each window (for the given signature options) and concatenate the results
    into a tensor of shape [N, num_sig_features * num_windows].
    Parameters
    ----------
    num_intervals: int, dimension of the transformed data (default 8)
    """

    def __init__(
        self,
        window_name=None,
        window_depth=None,
        window_length=None,
        window_step=None,
        sig_tfm=None,
        sig_depth=None,
        rescaling=None,
    ):
        super().__init__()
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.sig_tfm = sig_tfm
        self.depth = sig_depth
        self.rescaling = rescaling

        self.window = _window_getter(
            self.window_name, self.window_depth, self.window_length, self.window_step
        )

    def fit(self, data, labels=None):
        self._is_fitted = True
        return self

    def transform(self, data):
        # Input checks
        self.check_is_fitted()

        # Path rescaling
        if self.rescaling == "pre":
            data = _rescale_path(data, self.depth)

        # Prepare for signature computation
        if self.sig_tfm == "signature":

            def transform(x):
                return esig.stream2sig(x, self.depth)[1:].reshape(-1, 1)

        else:

            def transform(x):
                return esig.stream2logsig(x, self.depth).reshape(1, -1)

        length = data.shape[1]

        # Compute signatures in each window returning the grouped structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                # Signature computation step
                signature = np.stack(
                    [transform(x[window.start : window.end]) for x in data]
                ).reshape(data.shape[0], -1)
                # Rescale if specified
                if self.rescaling == "post":
                    signature = _rescale_signature(signature, data.shape[2], self.depth)

                signature_group.append(signature)
            signatures.append(signature_group)

        # We are currently not considering deep models and so return all the
        # features concatenated together
        signatures = np.concatenate([x for lst in signatures for x in lst], axis=1)

        return signatures
