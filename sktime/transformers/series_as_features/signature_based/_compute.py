# -*- coding: utf-8 -*-
"""
compute.py
=======================
Class for signature computation over windows.
"""
import torch
import signatory
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.signature_based._window import window_getter
from sktime.transformers.series_as_features.signature_based._rescaling import (
    rescale_path,
    rescale_signature,
)


class _WindowSignatureTransform(BaseSeriesAsFeaturesTransformer):
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

        self.window = window_getter(
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
            data = rescale_path(data, self.depth)

        # Prepare for signature computation
        path_obj = signatory.Path(data, self.depth)
        transform = getattr(path_obj, self.sig_tfm)
        length = path_obj.size(1)

        # Compute signatures in each window returning the grouped structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                # Signature computation step
                signature = transform(window.start, window.end)
                # Rescale if specified
                if self.rescaling == "post":
                    signature = rescale_signature(signature, data.size(2), self.depth)

                signature_group.append(signature)
            signatures.append(signature_group)

        # We are currently not considering deep models and so return all the
        # features concatenated together
        signatures = torch.cat([x for lst in signatures for x in lst], axis=1)

        return signatures
