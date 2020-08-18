"""
signature_method.py
============================
Implementation of a SignatureClassifier that utilises the signature method of
feature extraction. This method was built according to the best practices
and methodologies described in the paper:
    "A Generalised Signature Method for Time Series"
    [arxiv](https://arxiv.org/pdf/2006.00873.pdf)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sktime.classification.base import BaseClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sktime.transformers.series_as_features.signature_based import \
    GeneralisedSignatureMethod
from sktime.transformers.series_as_features.signature_based._checks import \
    handle_sktime_signatures


class SignatureClassifier(BaseClassifier):
    """Classification module using signature-based features.

    This simply initialises the GeneralisedSignatureMethod class which builds
    the feature extraction pipeline, then creates a new pipeline by
    appending a classifier after the feature extraction step.

    The default parameters are set to best practice parameters found "A
    Generalised Signature Method for Time-Series":
        [https://arxiv.org/pdf/2006.00873.pdf]
    Note that the final classifier used on the UEA datasets involved tuning
    the hyperparameters:
        - `depth` over [1, 2, 3, 4, 5, 6]
        - `window_depth` over [2, 3, 4]
        - RandomForestClassifier hyper-paramters.
    as these were found to be the most dataset dependent hyper-parameters.
    Thus, we recommend always tuning *at least* these parameters to any given
    dataset.


    Parameters
    ----------
    classifier: sklearn estimator, This should be any sklearn-type
        estimator. Defaults to RandomForestClassifier.
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
    random_state: int, Random state initialisation.

    Attributes
    ----------------
    signature_method: sklearn.Pipeline, An sklearn pipeline that performs the
        signature feature extraction step.
    pipeline: sklearn.Pipeline, The classifier appended to the
        `signature_method` pipeline to make a classification pipeline.
    """
    def __init__(self,
                 classifier=None,
                 scaling='stdsc',
                 augmentation_list=('basepoint', 'addtime'),
                 window_name='dyadic',
                 window_depth=3,
                 window_length=None,
                 window_step=None,
                 rescaling=None,
                 sig_tfm='signature',
                 depth=4,
                 random_state=None,
                 ):
        super(SignatureClassifier, self).__init__()
        self.classifier = classifier
        self.scaling = scaling
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

        self.signature_method = GeneralisedSignatureMethod(scaling,
                                                           augmentation_list,
                                                           window_name,
                                                           window_depth,
                                                           window_length,
                                                           window_step,
                                                           rescaling,
                                                           sig_tfm,
                                                           depth,
                                                           ).signature_method

    def setup_classification_pipeline(self):
        """ Setup the full signature method pipeline. """
        # Use rf if no classifier is set
        if self.classifier is None:
            classifier = RandomForestClassifier(
                random_state=self.random_state
            )
        else:
            classifier = self.classifier

        # Main classification pipeline
        self.pipeline = Pipeline([
            ('signature_method', self.signature_method),
            ('classifier', classifier)
        ])

    # Handle the sktime fit checks and convert to a tensor
    @handle_sktime_signatures(check_fitted=False)
    def fit(self, data, labels):
        # Join the classifier onto the signature method pipeline
        self.setup_classification_pipeline()

        # Fit the pre-initialised classification pipeline
        self.pipeline.fit(data, labels)
        self._is_fitted = True
        return self

    # Handle the sktime predict checks and convert to tensor format
    @handle_sktime_signatures(check_fitted=True)
    def predict(self, data):
        return self.pipeline.predict(data)

    # Handle the sktime predict checks and convert to tensor format
    @handle_sktime_signatures(check_fitted=True)
    def predict_proba(self, data):
        return self.pipeline.predict_proba(data)


def basic_signature_hyperopt(X,
                             y,
                             cv=5,
                             n_iter=10,
                             return_gs=False,
                             random_state=None):
    """Performs the hyperparameter search that is seen in "A Generalised
    Signature Method for Time Series.

    This follows from the work seen in "A Generalised Signature Method for
    Time Series" that found for most problems a suitable 'best practices'
    model should have:
         classifier: RandomForestClassifier.
         classifier_kwargs: Selected via gridsearch on standard random forest
            hyperparameters.
         scaling: Standard scaling.
         augmentation_list: ['basepoint', 'addtime'].
         window_name: 'dyadic'.
         window_kwargs: Search over dyadic depth [2, 3, 4].
         rescaling: 'post'.
         sig_tfm: 'signature'.
         depth: Hyper-parameter search over [1, 2, 3, 4, 5, 6].

    Parameters
    ----------
    X: pd.DataFrame, Sktime formatted training data.
    y: pd.DataFrame, Corresponding sktime formatted training labels.
    cv: int/sklearn.cv, A sklearn CV object or an integer specifying the number
        of splits that will be passed to a StratifiedKFoldCV method.
    n_iter: int, Number of iterations for the random gridsearch.
    return_gs: bool, Set True to return the full gridsearch object,
        otherwise will return just the best classifier.
    random_state: int, Sets the random state.
    """
    np.random.seed(random_state)
    signature_grid = {
        # Signature params
        'scaling': ['stdsc'],
        'depth': [1, 2, 3, 4, 5, 6],
        'window_name': ['dyadic'],
        'augmentation_list': [['basepoint', 'addtime']],
        'window_depth': [1, 2, 3, 4],
        'rescaling': ['post'],
        'random_state': [random_state],

        # Classifier and classifier params
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 500, 1000],
        'classifier__max_depth': [2, 4, 6, 8, 12, 16, 24, 32, 45, 60, 80, 100],
        'classifier__random_state': [random_state],

    }

    # Setup cv
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv)

    # Initialise the estimator
    estimator = SignatureClassifier()

    # Run a random grid search and return the gs object
    gs = RandomizedSearchCV(estimator, signature_grid, cv=cv, n_iter=n_iter,
                            random_state=random_state)
    gs.fit(X, y)

    out = gs if return_gs else gs.best_estimator_

    return out
