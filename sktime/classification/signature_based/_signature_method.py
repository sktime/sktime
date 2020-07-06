"""
signature_method.py
============================
Implementation of a SignatureClassifier that utilises the signature method of feature extraction. This method was built
according to the best practices and methodologies described in the paper:
    "A Generalised Signature Method for Time Series" - [arxiv](https://arxiv.org/pdf/2006.00873.pdf)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sktime.transformers.series_as_features.signature_based import GeneralisedSignatureMethod
from sktime.transformers.series_as_features.signature_based._checks import handle_sktime_signatures


class SignatureClassifier(GeneralisedSignatureMethod, BaseEstimator, ClassifierMixin):
    """Classification module using signature-based features.

    This simply initialises the GeneralisedSignatureMethod class which builds the feature extraction pipeline, then
    creates a new pipeline by appending a classifier after the feature extraction step.

    Parameters
    ----------
    classifier: sklearn estimator, This should be any sklearn-type estimator.

    Other Parameters
    ----------------
    See GeneralisedSignatureMethod parameters.
    """
    def __init__(self,
                 classifier,
                 scaling='stdsc',
                 augmentation_list=['basepoint', 'addtime'],
                 window_name='dyadic',
                 window_kwargs={'depth': 3},
                 rescaling=None,
                 sig_tfm='signature',
                 depth=4,
                 ):
        super(SignatureClassifier, self).__init__(scaling,
                                                  augmentation_list,
                                                  window_name,
                                                  window_kwargs,
                                                  rescaling,
                                                  sig_tfm,
                                                  depth
                                                  )
        self.classifier = classifier

        # Ready a classifier and join the signature method and classifier into a pipeline.
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
        return self.pipeline.predict_proba(data)

    @handle_sktime_signatures(check_fitted=True)
    def predict_proba(self, data):
        return self.pipeline.predict_proba(data)


if __name__ == '__main__':
    from sktime.utils.load_data import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe("../../../sktime/datasets/data/GunPoint/GunPoint_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe("../../../sktime/datasets/data/GunPoint/GunPoint_TEST.ts")
    classifier = RandomForestClassifier()
    signature_pipeline = SignatureClassifier(classifier).fit(train_x, train_y)
    signature_pipeline.predict(test_x)


