# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
from sktime.transformers.base import _PanelToTabularTransformer
from sktime.transformers.panel.signature_based._compute import (
    _WindowSignatureTransform,
)
from sktime.transformers.panel.signature_based._augmentations import (
    _make_augmentation_pipeline,
)
from sktime.transformers.panel.signature_based._checks import (
    _handle_sktime_signatures,
)
from sktime.utils.check_imports import _check_soft_dependencies

_check_soft_dependencies("esig")


class GeneralisedSignatureMethod(_PanelToTabularTransformer):
    """The generalised signature method of feature extraction.


    Parameters
    ----------
    augmentation_list: list of tuple of strings, List of augmentations to be
        applied before the signature transform is applied.
    window_name: str, The name of the window transform to apply.
    window_depth: int, The depth of the dyadic window. (Active only if
        `window_name == 'dyadic']`.
    window_length: int, The length of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding'].
    window_step: int, The step of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding'].
    rescaling: str, The method of signature rescaling.
    sig_tfm: str, String to specify the type of signature transform. One of:
        ['signature', 'logsignature']).
    depth: int, Signature truncation depth.

    Attributes
    ----------
    signature_method: sklearn.Pipeline, A sklearn pipeline object that contains
        all the steps to extract the signature features.
    """

    def __init__(
        self,
        augmentation_list=("basepoint", "addtime"),
        window_name="dyadic",
        window_depth=3,
        window_length=None,
        window_step=None,
        rescaling=None,
        sig_tfm="signature",
        depth=4,
    ):
        super(GeneralisedSignatureMethod, self).__init__()
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth

        self.setup_feature_pipeline()

    def _assertions(self):
        """ Some assertions to run on initialisation. """
        assert not all([self.sig_tfm == "logsignature", self.rescaling == "post"]), (
            "Cannot have post rescaling with the " "logsignature."
        )

    def setup_feature_pipeline(self):
        """ Sets up the signature method as an sklearn pipeline. """
        augmentation_step = _make_augmentation_pipeline(self.augmentation_list)
        transform_step = _WindowSignatureTransform(
            window_name=self.window_name,
            window_depth=self.window_depth,
            window_length=self.window_length,
            window_step=self.window_step,
            sig_tfm=self.sig_tfm,
            sig_depth=self.depth,
            rescaling=self.rescaling,
        )

        # The so-called 'signature method' as defined in the reference paper
        self.signature_method = Pipeline(
            [
                ("augmentations", augmentation_step),
                ("window_and_transform", transform_step),
            ]
        )

    @_handle_sktime_signatures(check_fitted=False)
    def fit(self, data, labels=None):
        self.signature_method.fit(data, labels)
        self._is_fitted = True
        return self

    @_handle_sktime_signatures(check_fitted=True)
    def transform(self, data, labels=None):
        return self.signature_method.transform(data)
