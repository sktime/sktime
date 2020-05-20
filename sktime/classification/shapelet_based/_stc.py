""" Shapelet Transform Classifier
wrapper implementation of a shapelet transform classifier pipeline that
simply performs a (configurable) shapelet transform
then builds (by default) a random forest. This is a stripped down version
for basic usage

"""

__author__ = "Tony Bagnall"
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import class_distribution
from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.shapelets import \
    ContractedShapeletTransform
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y


class ShapeletTransformClassifier(BaseClassifier):
    """ Shapelet Transform Classifier
        Basic implementation along the lines of
    @article{hills14shapelet,
      title={Classification of time series by shapelet transformation},
      author={J. Hills  and  J. Lines and E. Baranauskas and J. Mapp and A.
      Bagnall},
      journal={Data Mining and Knowledge Discovery},
      volume={28},
      number={4},
      pages={851--881},
      year={2014}
    }
    but with some of the refinements presented in
    @article{bostrom17binary,
      author={A. Bostrom and A. Bagnall},
      title={Binary Shapelet Transform for Multiclass Time Series
      Classification},
      journal={Transactions on Large-Scale Data and Knowledge Centered
      Systems},
      volume={32},
      year={2017},
      pages={24--46}
    }
    """

    def __init__(
            self,
            time_contract_in_mins=300,
            n_estimators=500
    ):
        self.time_contract_in_mins = time_contract_in_mins
        self.n_estimators = n_estimators

        self.classifier = Pipeline([
            ('st', ContractedShapeletTransform(
                time_contract_in_mins=time_contract_in_mins,
                verbose=False)),
            ('rf', RandomForestClassifier(n_estimators=n_estimators))
        ])

        #        self.shapelet_transform=ContractedShapeletTransform(
        #        time_limit_in_mins=self.time_contract_in_mins, verbose=shouty)
        #        self.classifier=RandomForestClassifier(
        #        n_estimators=self.n_estimators,criterion="entropy")
        #        self.st_X=None;
        super(ShapeletTransformClassifier, self).__init__()

    def fit(self, X, y):
        """Perform a shapelet transform then builds a random forest.
        Contract default for ST is 5 hours
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. RISE has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
         """
        X, y = check_X_y(X, y, enforce_univariate=True)
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        self.classifier.fit(X, y)

        #        self.shapelet_transform.fit(X,y)
        #        print("Shapelet Search complete")
        #        self.st_X =self.shapelet_transform.transform(X)
        #        print("Transform complete")
        #        X = np.asarray([a.values for a in X.iloc[:, 0]])
        #        self.classifier.fit(X,y)
        #       print("Build classifier complete")
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts] or a
        data frame.
        If a Pandas data frame is passed,

        Returns
        -------
        output : array of shape = [n_samples]
        """
        probs = self.predict_proba(X)
        return np.array([self.classes_[np.argmax(prob)] for prob in probs])

    def predict_proba(self, X):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,

        Returns
        -------
        output : array of shape = [n_samples, num_classes] of probabilities
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)
        return self.classifier.predict_proba(X)
