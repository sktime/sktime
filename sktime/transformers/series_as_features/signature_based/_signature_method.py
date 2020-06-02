import torch
from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.signature_based._compute import WindowSignatureTransform
from sktime.transformers.series_as_features.signature_based._augmentations import get_augmentation_pipeline
from sktime.transformers.series_as_features.signature_based._rescaling import TrickScaler
from sktime.utils.validation.series_as_features import check_X_y, check_X
from sktime.utils.data_container import nested_to_3d_numpy


class GeneralisedSignatureMethod(BaseSeriesAsFeaturesTransformer):
    """The generalised signature method of feature extraction.

    [https://arxiv.org/pdf/2006.00873.pdf]
    """
    def __init__(self,
                 scaling='stdsc',
                 augmentation_list=['basepoint', 'addtime'],
                 window_name='dyadic',
                 window_kwargs={'depth': 3},
                 sig_tfm='signature',
                 depth=4,
                 ):
        self.scaling = scaling
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.sig_tfm = sig_tfm
        self.depth = depth

        self.setup_pipeline()

    def setup_pipeline(self):
        """ Sets up the signature method as an sklearn pipeline given model options. """
        scaling_step = TrickScaler(scaling=self.scaling)
        augmentation_step = get_augmentation_pipeline(self.augmentation_list)
        transform_step = WindowSignatureTransform(self.sig_tfm, self.depth, self.window_name, self.window_kwargs)

        # The so-called 'signature method' as defined in the reference paper
        self.signature_method = Pipeline([
            ('scaling', scaling_step),
            ('augmentations', augmentation_step),
            ('window_and_transform', transform_step),
        ])

    def fit(self, data, labels=None):
        data, labels = check_X_y(data, labels, enforce_univariate=False)
        tensor_data = torch.Tensor(nested_to_3d_numpy(data)).transpose(1, 2)

        self.signature_method.fit(tensor_data, labels)
        self._is_fitted = True
        return self

    def transform(self, data):
        # sktime checks
        self.check_is_fitted()
        data = check_X(data, enforce_univariate=False)

        # Signature functionality requires torch tensors
        tensor_data = torch.Tensor(nested_to_3d_numpy(data)).transpose(1, 2)

        return self.signature_method.transform(tensor_data)


