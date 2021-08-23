# -*- coding: utf-8 -*-
"""Implementation of a SignatureClassifier.

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
        "A Generalised Signature Method for Multivariate TimeSeries" [1]

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
    classifier : sklearn estimator, default=RandomForestClassifier
        This should be any sklearn-type estimator. Defaults to RandomForestClassifier.
    augmentation_list: list of tuple of strings, default=("basepoint", "addtime")
        List of augmentations to be applied before the signature transform is applied.
    window_name: str, default="dyadic"
        The name of the window transform to apply.
    window_depth: int, default=3
        The depth of the dyadic window. (Active only if `window_name == 'dyadic']`.
    window_length: int, default=None
        The length of the sliding/expanding window. (Active only if `window_name in
        ['sliding, 'expanding'].
    window_step: int, default=None
        The step of the sliding/expanding window. (Active only if `window_name in
        ['sliding, 'expanding'].
    rescaling: str, default=None
        The method of signature rescaling.
    sig_tfm: str, default="signature"
        String to specify the type of signature transform. One of:
        ['signature', 'logsignature']).
    depth: int, default=4
        Signature truncation depth.
    random_state: int, default=None
        Random state initialisation.

    Attributes
    ----------
    signature_method: sklearn.Pipeline
        An sklearn pipeline that performs the signature feature extraction step.
    pipeline: sklearn.Pipeline
        The classifier appended to the `signature_method` pipeline to make a
        classification pipeline.

    References
    ----------
    .. [1] Morrill, James, et al. "A generalised signature method for multivariate time
        series feature extraction." arXiv preprint arXiv:2006.00873 (2020).
        https://arxiv.org/pdf/2006.00873.pdf

    See Also
    --------
    SignatureTransformer

    Examples
    --------
    >>> from sktime.classification.feature_based import SignatureClassifier
    >>> from sktime.datasets import load_italy_power_demand
    >>> X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    >>> X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    >>> clf = SignatureClassifier()
    >>> clf.fit(X_train, y_train)
    SignatureClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

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
        self.pipeline = None
        self.classes_ = []

    def _setup_classification_pipeline(self):
        """Set up the full signature method pipeline."""
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
        """Fit an estimator using transformed data from the SignatureTransformer.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_dims]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        self.classes_ = np.unique(labels)
        # Join the classifier onto the signature method pipeline
        self._setup_classification_pipeline()

        # Fit the pre-initialised classification pipeline
        self.pipeline.fit(data, labels)
        self._is_fitted = True
        return self

    # Handle the sktime predict checks and convert to tensor format
    @_handle_sktime_signatures(check_fitted=True, force_numpy=True)
    def predict(self, data):
        """Predict class values of n_instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, n_dims)

        Returns
        -------
        preds : np.ndarray of shape (n, 1)
            Predicted class.
        """
        return self.pipeline.predict(data)

    # Handle the sktime predict checks and convert to tensor format
    @_handle_sktime_signatures(check_fitted=True, force_numpy=True)
    def predict_proba(self, data):
        """Predict class probabilities for n_instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, n_dims)

        Returns
        -------
        predicted_probs : array of shape (n_instances, n_classes)
            Predicted probability of each class.
        """
        return self.pipeline.predict_proba(data)
