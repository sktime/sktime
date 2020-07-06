from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.signature_based._compute import \
    WindowSignatureTransform
from sktime.transformers.series_as_features.signature_based._augmentations \
    import get_augmentation_pipeline
from sktime.transformers.series_as_features.signature_based._rescaling import \
    TrickScaler
from sktime.transformers.series_as_features.signature_based._checks import \
    handle_sktime_signatures


class GeneralisedSignatureMethod(BaseSeriesAsFeaturesTransformer):
    """The generalised signature method of feature extraction.

    This follows the methodologies and best practices described in "M
    [https://arxiv.org/pdf/2006.00873.pdf]

    Parameters
    ----------
    scaling: str, Method of scaling.
    augmentation_list: list of strings, List of augmentations to be applied
        before the signature transform is applied.
    window_name: str, The name of the window transform to apply.
    window_kwargs: dict, Additional parameters to be supplied to the window
        method.
    rescaling: str, The method of signature rescaling.
    sig_tfm: str, String to specify the type of signature transform. One of:
        ['signature', 'logsignature']).
    depth: int, Signature truncation depth.

    Attributes
    ----------
    signature_method: sklearn.Pipeline, A sklearn pipeline object that contains
        all the steps to extract the signature features.
    """
    def __init__(self,
                 scaling='stdsc',
                 augmentation_list=['basepoint', 'addtime'],
                 window_name='dyadic',
                 window_kwargs={'depth': 3},
                 rescaling=None,
                 sig_tfm='signature',
                 depth=4,
                 ):
        self.scaling = scaling
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth

        self.setup_feature_pipeline()

    def setup_feature_pipeline(self):
        """ Sets up the signature method as an sklearn pipeline. """
        scaling_step = TrickScaler(scaling=self.scaling)
        augmentation_step = get_augmentation_pipeline(self.augmentation_list)
        transform_step = WindowSignatureTransform(
            self.window_name, self.window_kwargs, self.sig_tfm, self.depth,
            rescaling=self.rescaling
        )

        # The so-called 'signature method' as defined in the reference paper
        self.signature_method = Pipeline([
            ('scaling', scaling_step),
            ('augmentations', augmentation_step),
            ('window_and_transform', transform_step),
        ])

    @handle_sktime_signatures(check_fitted=False)
    def fit(self, data, labels=None):
        self.signature_method.fit(data, labels)
        self._is_fitted = True
        return self

    @handle_sktime_signatures(check_fitted=True)
    def transform(self, data):
        return self.signature_method.transform(data)
