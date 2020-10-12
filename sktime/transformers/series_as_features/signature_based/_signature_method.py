from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.signature_based._compute import (
    _WindowSignatureTransform,
)
from sktime.transformers.series_as_features.signature_based._augmentations  \
    import make_augmentation_pipeline
from sktime.transformers.series_as_features.signature_based._rescaling import (
    TrickScaler,
)
from sktime.transformers.series_as_features.signature_based._checks import (
    handle_sktime_signatures,
)


class GeneralisedSignatureMethod(BaseSeriesAsFeaturesTransformer):
    """The generalised signature method of feature extraction.

    Parameters
    ----------
    scaling: str, Method of scaling.
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
        scaling="stdsc",
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
        self.scaling = scaling
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth

        self.setup_feature_pipeline()

    def setup_feature_pipeline(self):
        """ Sets up the signature method as an sklearn pipeline. """
        scaling_step = TrickScaler(scaling=self.scaling)
        augmentation_step = make_augmentation_pipeline(self.augmentation_list)
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
                ("scaling", scaling_step),
                ("augmentations", augmentation_step),
                ("window_and_transform", transform_step),
            ]
        )

    @handle_sktime_signatures(check_fitted=False)
    def fit(self, data, labels=None):
        self.signature_method.fit(data, labels)
        self._is_fitted = True
        return self

    @handle_sktime_signatures(check_fitted=True)
    def transform(self, data, labels=None):
        return self.signature_method.transform(data)
