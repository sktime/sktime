"""
compute.py
=======================
Class for signature computation over windows.
"""
import torch
import signatory
from sklearn.base import TransformerMixin
from sktime.transformers.series_as_features.signature_based.window import window_getter


class WindowSignatureTransform(TransformerMixin):
    """Performs the signature transform over given windows.

    Given data of shape [N, L, C] and specification of a window method from the signatures window module, this class
    will compute the signatures over each window (for the given signature options) and concatenate the results into a
    tensor of shape [N, num_sig_features * num_windows].
    """
    def __init__(self, sig_tfm, depth, window_name, window_kwargs):
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.window = window_getter(self.window_name, **self.window_kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Prepare for signature computation
        path_obj = signatory.Path(X, self.depth)
        transform = getattr(path_obj, self.sig_tfm)
        length = path_obj.size(1)

        # Compute signatures in each window returning the grouped list structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                signature = transform(window.start, window.end)
                signature_group.append(signature)
            signatures.append(signature_group)

        # We are currently not considering deep models and so return all the features concatenated together
        signatures = torch.cat([x for l in signatures for x in l], axis=1)

        return signatures


