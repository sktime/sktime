""" Shapelet Transform Classifier
wrapper implementation of a shapelet transform classifier pipeline that simply performs a (configurable) shapelet transform
then builds (by default) a random forest. This is a stripped down version for basic usage

"""

__author__ = "Tony Bagnall"
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import class_distribution

from sktime.classifiers.base import BaseClassifier
from sktime.transformers.shapelets import ContractedShapeletTransform
from sktime.utils.validation.supervised import validate_X_y, validate_X


class ShapeletTransformClassifier(BaseClassifier):
    """ Shapelet Transform Classifier
        Basic implementation along the lines of
    @article{hills14shapelet,
      title={Classification of time series by shapelet transformation},
      author={J. Hills  and  J. Lines and E. Baranauskas and J. Mapp and A. Bagnall},
      journal={Data Mining and Knowledge Discovery},
      volume={28},
      number={4},
      pages={851--881},
      year={2014}
    }
    but with some of the refinements presented in
    @article{bostrom17binary,
      author={A. Bostrom and A. Bagnall},
      title={Binary Shapelet Transform for Multiclass Time Series Classification},
      journal={Transactions on Large-Scale Data and Knowledge Centered Systems},
      volume={32},
      year={2017},
      pages={24--46}
    }


    """

    def __init__(self, time_contract_in_mins=300, n_classifiers=500, random_state=None):
        self.time_contract_in_mins = time_contract_in_mins
        self.n_classifiers = n_classifiers
        self.random_state = random_state

        self.transform = ContractedShapeletTransform(
            time_limit_in_mins=time_contract_in_mins,
            random_state=random_state,
            verbose=False
        )

        self.internal_classifier = RandomForestClassifier(
            n_estimators=n_classifiers,
            random_state=random_state
        )

        self.pipeline = Pipeline([
            ('st', self.transform),
            ('rf', self.internal_classifier)
        ])

    #        self.shapelet_transform=ContractedShapeletTransform(time_limit_in_mins=self.time_contract_in_mins, verbose=shouty)
    #        self.classifier=RandomForestClassifier( n_estimators=self.n_classifiers,criterion="entropy")
    #        self.st_X=None;

    def fit(self, X, y):
        """Perform a shapelet transform then builds a random forest. Contract default for ST is 5 hours
        ----------
        X : array-like or sparse matrix of shape = [n_instances,series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it must have a single column (i.e. univariate
            classification. RISE has no bespoke method for multivariate classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
         """

        validate_X_y(X, y)
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        self.pipeline.fit(X, y)

        #        self.shapelet_transform.fit(X,y)
        #        print("Shapelet Search complete")
        #        self.st_X =self.shapelet_transform.transform(X)
        #        print("Transform complete")
        #        X = np.asarray([a.values for a in X.iloc[:, 0]])
        #        self.classifier.fit(X,y)
        #       print("Build classifier complete")
        return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts] or a data frame.
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

        -------
        output : array of shape = [n_samples, num_classes] of probabilities
        """
        #        tempX=self.shapelet_transform.transform(X)
        #        X = np.asarray([a.values for a in tempX.iloc[:, 0]])
        validate_X(X)
        return self.pipeline.predict_proba(X)

    #
    # def set_contract_minutes(self, minutes):
    #     self.time_contract_in_mins = minutes
    #     self.shapelet_transform.time_limit_in_mins = minutes
    #
    # def set_classifier(self, cls):
    #     self.classifier = cls
