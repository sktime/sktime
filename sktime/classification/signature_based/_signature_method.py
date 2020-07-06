"""
signature_method.py
============================
Implementation of a SignatureClassifier that utilises the signature method of
feature extraction. This method was built according to the best practices
and methodologies described in the paper:
    "A Generalised Signature Method for Time Series"
    [arxiv](https://arxiv.org/pdf/2006.00873.pdf)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
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

    Parameters
    ----------
    classifier: sklearn estimator, This should be any sklearn-type
        estimator. This defaults to RandomForestClassifier if left as None.

    Other Parameters
    ----------------
    See GeneralisedSignatureMethod parameters.
    """
    def __init__(self,
                 classifier=None,
                 scaling='stdsc',
                 augmentation_list=['basepoint', 'addtime'],
                 window_name='dyadic',
                 window_kwargs={'depth': 3},
                 rescaling=None,
                 sig_tfm='signature',
                 depth=4,
                 random_state=None
                 ):
        super(SignatureClassifier, self).__init__()
        self.scaling = scaling
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.classifier = RandomForestClassifier() if classifier is None \
            else classifier
        self.random_state = random_state

        self.signature_method = GeneralisedSignatureMethod(scaling,
                                                           augmentation_list,
                                                           window_name,
                                                           window_kwargs,
                                                           rescaling,
                                                           sig_tfm,
                                                           depth
                                                           ).signature_method
        # Ready a classifier and join the signature method and classifier into
        # a pipeline.
        self.setup_classification_pipeline()

    def setup_classification_pipeline(self):
        """ Setup the full signature method pipeline. """
        self.pipeline = Pipeline([
            ('signature_method', self.signature_method),
            ('classifier', self.classifier)
        ])

    @handle_sktime_signatures(check_fitted=False)
    def fit(self, data, labels):
        # Fit the pre-initialised classification pipeline
        self.pipeline.fit(data, labels)
        self._is_fitted = True
        return self

    @handle_sktime_signatures(check_fitted=True)
    def predict(self, data):
        return self.pipeline.predict(data)

    @handle_sktime_signatures(check_fitted=True)
    def predict_proba(self, data):
        return self.pipeline.predict_proba(data)


def example_signature_hyperparameter_search(train_data,
                                            train_labels,
                                            n_splits=5,
                                            n_iter=10):
    """A simple hyper-parameter search to evaluate the best params from the
    SignatureClassifier.

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
    train_data: pd.DataFrame, sktime formatted training data.
    train_labels: pd.DataFrame, Corresponding sktime formatted training labels.
    n_splits: int, The number of cross-validation folds to use. Splits the data
        using a StratifiedKFoldCV method.
    n_iter: int, Number of iterations for the random gridsearch.

    Returns
    -------
    SignatureClassifier: A trained signature classifier that achieved the best
        accuracy on the validation folds.
    """
    classifier = RandomForestClassifier()

    signature_grid = {
        # Signature params
        'scaling': ['stdsc'],
        'depth': [1, 2, 3, 4, 5, 6],
        'window_name': ['dyadic'],
        'augmentation_list': [['basepoint', 'addtime']],
        'window_kwargs': [
            {'depth': 1},
            {'depth': 2},
            {'depth': 3},
            {'depth': 4},
        ],
        'rescaling': ['post'],

        # Classifier and classifier params
        'classifier': [classifier],
        'classifier__n_estimators': [50, 100, 500, 1000],
        'classifier__max_depth': [2, 4, 6, 8, 12, 16, 24, 32, 45, 60, 80, 100]
    }

    # Initialise the estimator
    estimator = SignatureClassifier(classifier=classifier)

    # Run a random grid search and return the gs object
    cv = StratifiedKFold(n_splits=n_splits)
    gs = RandomizedSearchCV(estimator, signature_grid, cv=cv, n_iter=n_iter)
    gs.fit(train_data, train_labels)

    # Best classifier
    best_estimator = gs.best_estimator_
    return best_estimator


if __name__ == '__main__':
    from sktime.utils.load_data import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe("../../../sktime/datasets/data/BasicMotions/BasicMotions_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe("../../../sktime/datasets/data/BasicMotions/BasicMotions_TEST.ts")
    import torch
    train_y = torch.ones(train_y.shape).numpy()
    test_y = torch.ones(test_y.shape).numpy()
    self = SignatureClassifier().fit(train_x, train_y)
    self.predict(test_x)

