from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.signature_based.compute import WindowSignatureTransform
from sktime.transformers.series_as_features.signature_based.augmentations import get_augmentation_pipeline


class GeneralisedSignatureMethod(TransformerMixin):
    """The generalised signature method of feature extraction.

    [https://arxiv.org/pdf/2006.00873.pdf]
    """
    def __init__(self,
                 augmentation_list=['basepoint', 'addtime'],
                 window_name='dyadic',
                 window_kwargs={'depth': 3},
                 sig_tfm='signature',
                 depth=4,
                 ):
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.sig_tfm = sig_tfm
        self.depth = depth

        self.setup_pipeline()

    def setup_pipeline(self):
        """ Sets up the signature method as an sklearn pipeline given model options. """
        augmentation_step = get_augmentation_pipeline(self.augmentation_list)
        transform_step = WindowSignatureTransform(self.sig_tfm, self.depth, self.window_name, self.window_kwargs)

        # The so-called 'signature method' as defined in the reference paper
        self.signature_method = Pipeline([
            ('augmentations', augmentation_step),
            ('window_and_transform', transform_step),
        ])

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        return self.signature_method.transform(data)
