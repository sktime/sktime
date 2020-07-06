"""
compute.py
=======================
Class for signature computation over windows.
"""
import torch
import signatory
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.signature_based._window import window_getter
from sktime.transformers.series_as_features.signature_based._rescaling import rescale_path, rescale_signature


class WindowSignatureTransform(BaseSeriesAsFeaturesTransformer):
    """Performs the signature transform over given windows.

    Given data of shape [N, L, C] and specification of a window method from the signatures window module, this class
    will compute the signatures over each window (for the given signature options) and concatenate the results into a
    tensor of shape [N, num_sig_features * num_windows].

    Parameters
    ----------
    num_intervals  : int, dimension of the transformed data (default 8)
    """
    def __init__(self, window_name, window_kwargs, sig_tfm, depth, rescaling=None):
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.rescaling = rescaling

        self.window = window_getter(self.window_name, **self.window_kwargs)
        self.set_rescaling()

    def set_rescaling(self):
        # Setup rescaling options
        self.pre_rescaling = lambda path, depth: path
        self.post_rescaling = lambda signature, channels, depth: signature
        if self.rescaling == 'pre':
            self.pre_rescaling = rescale_path
        elif self.rescaling == 'post':
            self.post_rescaling = rescale_signature

    def transform(self, data):
        # Path rescaling
        data = self.pre_rescaling(data, self.depth)

        # Prepare for signature computation
        path_obj = signatory.Path(data, self.depth)
        transform = getattr(path_obj, self.sig_tfm)
        length = path_obj.size(1)

        # Compute signatures in each window returning the grouped list structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                # Signature computation step
                signature = transform(window.start, window.end)
                # Rescale if specified
                rescaled_signature = self.post_rescaling(signature, data.size(2), self.depth)

                signature_group.append(rescaled_signature)
            signatures.append(signature_group)

        # We are currently not considering deep models and so return all the features concatenated together
        signatures = torch.cat([x for l in signatures for x in l], axis=1)

        return signatures


