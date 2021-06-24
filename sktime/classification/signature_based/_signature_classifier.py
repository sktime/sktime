# -*- coding: utf-8 -*-
"""Implementation of a SignatureClassifier

Utilises the signature method of feature extraction.
This method was built according to the best practices
and methodologies described in the paper:
    "A Generalised Signature Method for Time Series"
    [arxiv](https://arxiv.org/pdf/2006.00873.pdf).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.signature_based._checks import (
    _handle_sktime_signatures,
)
from sktime.transformations.panel.signature_based._signature_method import (
    SignatureTransformer,
)


class SignatureClassifier(BaseClassifier):
    """Classification module using signature-based features.

    This simply initialises the SignatureTransformer class which builds
    the feature extraction pipeline, then creates a new pipeline by
    appending a classifier after the feature extraction step.

    The default parameters are set to best practice parameters found in
        "A Generalised Signature Method for Multivariate TimeSeries"
        [https://arxiv.org/pdf/2006.00873.pdf]

    Note that the final classifier used on the UEA datasets involved tuning
    the hyper-parameters:
        - `depth` over [1, 2, 3, 4, 5, 6]
        - `window_depth` over [2, 3, 4]
        - RandomForestClassifier hyper-parameters.
    as these were found to be the most dataset dependent hyper-parameters.

    Thus, we recommend always tuning *at least* these parameters to any given
    dataset.

    Parameters
    ----------
    classifier: sklearn estimator, This should be any sklearn-type
        estimator. Defaults to RandomForestClassifier.
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
    random_state: int, Random state initialisation.

    Attributes
    ----------------
    signature_method: sklearn.Pipeline, An sklearn pipeline that performs the
        signature feature extraction step.
    pipeline: sklearn.Pipeline, The classifier appended to the
        `signature_method` pipeline to make a classification pipeline.
    """

    def __init__(
        self,
        classifier=None,
        augmentation_list=("basepoint", "addtime"),
        window_name="dyadic",
        window_depth=3,
        window_length=None,
        window_step=None,
        rescaling=None,
        sig_tfm="signature",
        depth=4,
        random_state=None,
    ):
        super(SignatureClassifier, self).__init__()
        self.classifier = classifier
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.random_state = random_state
        np.random.seed(random_state)

        self.signature_method = SignatureTransformer(
            augmentation_list,
            window_name,
            window_depth,
            window_length,
            window_step,
            rescaling,
            sig_tfm,
            depth,
        ).signature_method

    def _setup_classification_pipeline(self):
        """Setup the full signature method pipeline."""
        # Use rf if no classifier is set
        if self.classifier is None:
            classifier = RandomForestClassifier(random_state=self.random_state)
        else:
            classifier = self.classifier

        # Main classification pipeline
        self.pipeline = Pipeline(
            [("signature_method", self.signature_method), ("classifier", classifier)]
        )

    # Handle the sktime fit checks and convert to a tensor
    @_handle_sktime_signatures(check_fitted=False)
    def fit(self, data, labels):
        # Join the classifier onto the signature method pipeline
        self._setup_classification_pipeline()

        # Fit the pre-initialised classification pipeline
        self.pipeline.fit(data, labels)
        self._is_fitted = True
        return self

    # Handle the sktime predict checks and convert to tensor format
    @_handle_sktime_signatures(check_fitted=True, force_numpy=True)
    def predict(self, data):
        return self.pipeline.predict(data)

    # Handle the sktime predict checks and convert to tensor format
    @_handle_sktime_signatures(check_fitted=True, force_numpy=True)
    def predict_proba(self, data):
        return self.pipeline.predict_proba(data)
