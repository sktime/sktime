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

from sktime.base._base import _clone_estimator
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
    estimator : sklearn estimator, default=RandomForestClassifier
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
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.

    References
    ----------
    .. [1] Morrill, James, et al. "A generalised signature method for multivariate time
        series feature extraction." arXiv preprint arXiv:2006.00873 (2020).
        https://arxiv.org/pdf/2006.00873.pdf

    See Also
    --------
    SignatureTransformer
    """

    _tags = {
        "capability:multivariate": True,
        "capability:predict_proba": True,
        "classifier_type": "feature",
        "python_dependencies": "esig",
        "python_version": "<3.10",
    }

    def __init__(
        self,
        estimator=None,
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
        self.estimator = estimator
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.random_state = random_state

        super().__init__()

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

    def _setup_classification_pipeline(self):
        """Set up the full signature method pipeline."""
        # Use rf if no classifier is set
        if self.estimator is None:
            classifier = RandomForestClassifier(random_state=self.random_state)
        else:
            classifier = _clone_estimator(self.estimator, self.random_state)

        # Main classification pipeline
        self.pipeline = Pipeline(
            [("signature_method", self.signature_method), ("classifier", classifier)]
        )

    # Handle the sktime fit checks and convert to a tensor
    @_handle_sktime_signatures(check_fitted=False)
    def _fit(self, X, y):
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
        # Join the classifier onto the signature method pipeline
        self._setup_classification_pipeline()

        # Fit the pre-initialised classification pipeline
        self.pipeline.fit(X, y)

        return self

    # Handle the sktime predict checks and convert to tensor format
    @_handle_sktime_signatures(check_fitted=True, force_numpy=True)
    def _predict(self, X) -> np.ndarray:
        """Predict class values of n_instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, n_dims)

        Returns
        -------
        preds : np.ndarray of shape (n, 1)
            Predicted class.
        """
        return self.pipeline.predict(X)

    # Handle the sktime predict checks and convert to tensor format
    @_handle_sktime_signatures(check_fitted=True, force_numpy=True)
    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n_instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, n_dims)

        Returns
        -------
        predicted_probs : array of shape (n_instances, n_classes)
            Predicted probability of each class.
        """
        return self.pipeline.predict_proba(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"estimator": RandomForestClassifier(n_estimators=10)}
        else:
            return {
                "estimator": RandomForestClassifier(n_estimators=2),
                "augmentation_list": ("basepoint", "addtime"),
                "depth": 1,
                "window_name": "global",
            }
