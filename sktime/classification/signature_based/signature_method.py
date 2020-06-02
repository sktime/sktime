"""
signature_method.py
============================
Implementation of a SignatureClassifier that utilises the signature method of feature extraction. This method was built
according to the best practices and methodologies described in the paper:
    "A Generalised Signature Method for Time Series" - [arxiv](https://arxiv.org/pdf/2006.00873.pdf)

# TODO Implement deep models.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.signature_based import GeneralisedSignatureMethod
from sktime.utils.validation.series_as_features import check_X


class SignatureClassifier(BaseEstimator, ClassifierMixin):
    """Module for classification using signature features.

    This simply sticks a classifier on the end of the GeneralisedSignatureTransform class as we wish to classify from
    the features extracted by 'the signature method'.
    """
    def __init__(self,
                 augmentation_list=['basepoint', 'addtime'],
                 window_name='dyadic',
                 window_kwargs={'depth': 3},
                 sig_tfm='signature',
                 depth=4,
                 classifier='default'
                 ):
        # Generalised signature method for feature extraction
        self.signature_method = GeneralisedSignatureMethod(
            augmentation_list=augmentation_list,
            window_name=window_name,
            window_kwargs=window_kwargs,
            sig_tfm=sig_tfm,
            depth=depth
        )

        # Ready a classifier and join the signature method and classifier into a pipeline.
        self.classifier = RandomForestClassifier() if classifier == 'default' else classifier
        self.setup_pipeline()

        # Add some methods
        self.predict = self.pipeline.predict
        self.predict_proba = self.pipeline.predict_proba

    def setup_pipeline(self):
        """ Setup the full signature method pipeline. """
        self.pipeline = Pipeline([
            ('signature_method', self.signature_method),
            ('classifier', self.classifier)
        ])

    def fit(self, data, labels):
        # sktime checks
        data = check_X(data, enforce_univariate=False)

        # Fit the pipeline
        self.pipeline.fit(data, labels)

        return self


if __name__ == '__main__':
    from sktime.utils.load_data import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe("../../../sktime/datasets/data/GunPoint/GunPoint_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe("../../../sktime/datasets/data/GunPoint/GunPoint_TEST.ts")
    classifier = SignatureClassifier().fit(train_x, train_y)


