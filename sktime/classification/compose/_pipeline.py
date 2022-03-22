# -*- coding: utf-8 -*-
"""Pipeline with a classifier."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
import numpy as np
from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline

__author__ = ["fkiraly"]
__all__ = ["ClassifierPipeline"]


class ClassifierPipeline(BaseClassifier, _HeterogenousMetaEstimator):
    """Pipeline of transformers and a classifier.

    The `ClassifierPipeline` compositor chains transformers and a single classifier.
    The pipeline is constructed with a list of sktime transformers, plus a classifier,
        i.e., estimators following the BaseTransformer resp BaseClassifier interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a classifier `clf`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `clf.fit` with `X` being the output of `trafo[N]`,
        and `y` identical with the input to `self.fit`
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clf.predict` on the output of `trafoN.transform`,
        and returning the output of `clf.predict`
    `predict_proba(X)` - result is of executing `trafo1.transform`, `trafo2.transform`,
        etc, with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clf.predict_proba` on the output of `trafoN.transform`,
        and returning the output of `clf.predict_proba`

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `ClassifierPipeline` can also be created by using the magic multiplication
        on any classifier, i.e., if `my_clf` inherits from `BaseClassifier`,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_clf`
            will result in the same object as  obtained from the constructor
            `ClassifierPipeline(classifier=my_clf, transformers=[my_trafo1, my_trafo2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    classifier : sktime classifier, i.e., estimator inheriting from BaseClassifier
        this is a "blueprint" classifier, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    classifier_ : sktime classifier, clone of classifier in `classifier`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`
    """

    _tags = {
        "X_inner_mtype": "pd-multiindex",  # which type do _fit/_predict accept
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    _required_parameters = ["classifier"]

    # no default tag values - these are set dynamically below

    def __init__(self, classifier, transformers):

        self.classifier = classifier
        self.classifier_ = clone(classifier)
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super(ClassifierPipeline, self).__init__()

        # can handle multivariate of both classifier and all transformers can
        multivariate = classifier.get_tag("capability:multivariate", False)
        multivariate = multivariate and not self.transformers_.get_tag(
            "univariate-only", True
        )
        # can handle missing values if both classifier and all transformers can
        missing = classifier.get_tag("capability:missing_values", False)
        missing = missing and self.transformer_.get_tag("handles-missing-data", False)
        # can handle unequal length if classifier can
        #   transformers should always be able to, due to vectorization
        unequal = classifier.get_tag("capability:unequal_length")
        # last three tags are always False, since not supported by transformers
        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
            "capability:contractable": False,
            "capability:train_estimate": False,
            "capability:multithreading": False,
        }
        self.set_tags(**tags_to_set)

    @property
    def _transformers(self):
        return self.transformers_._steps

    @_transformers.setter
    def _transformers(self, value):
        self.transformers_._steps = value

    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a ClassifierPipeline
            new_pipeline = ClassifierPipeline(
                classifier=self.classifier,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    @staticmethod
    def _is_name_and_trafo(obj):
        if not isinstance(obj, tuple) or len(obj) != 2:
            return False
        if not isinstance(obj[0], str) or not isinstance(obj[1], BaseTransformer):
            return False
        return True

    def _anytagis(self, tag_name, value):
        """Return whether any estimator in list has tag `tag_name` of value `value`."""
        tagis = [est.get_tag(tag_name, value) == value for _, est in self.transformers_]
        return any(tagis)

    def _anytagis_then_set(self, tag_name, value, value_if_not):
        """Set self's `tag_name` tag to `value` if any estimator on the list has it."""
        if self._anytagis(tag_name=tag_name, value=value):
            self.set_tags(**{tag_name: value})
        else:
            self.set_tags(**{tag_name: value_if_not})

    def _anytag_notnone_val(self, tag_name):
        """Return first non-'None' value of tag `tag_name` in estimator list."""
        for _, est in self.transformers_:
            tag_val = est.get_tag(tag_name)
            if tag_val != "None":
                return tag_val
        return tag_val

    def _anytag_notnone_set(self, tag_name):
        """Set self's `tag_name` tag to first non-'None' value in estimator list."""
        tag_val = self._anytag_notnone_val(tag_name=tag_name)
        if tag_val != "None":
            self.set_tags(**{tag_name: tag_val})

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        core logic

        Parameters
        ----------
        X : Training data of type self.get_tag("X_inner_mtype")
        y : array-like, shape = [n_instances] - the class labels

        Returns
        -------
        self : reference to self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        Xt = self.transformers_.fit_transform(X)
        self.classifier_.fit(Xt, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        core logic

        Parameters
        ----------
        X : data not used in training, of type self.get_tag("X_inner_mtype")

        Returns
        -------
        y : predictions of labels for X, np.ndarray
        """
        Xt = self.transformers_.transform(X)
        return self.classifier_.predict(Xt)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : data to predict y with, of type self.get_tag("X_inner_mtype")

        Returns
        -------
        y : predictions of probabilities for class values of X, np.ndarray
        """
        Xt = self.transformers_.transform(X)
        return self.classifier_.predict_proba(Xt)

    def get_params(self, deep=True):
        """Get parameters of estimator in `transformers`.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()
        trafo_params = self._get_params("_transformers", deep=deep)
        params.update(trafo_params)

        return params

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "classifier" in kwargs.keys():
            if not isinstance(kwargs["classifier"], BaseClassifier):
                raise TypeError('"classifier" arg must be an sktime classifier')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        classif_keys = self.classifier.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        classif_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=classif_keys)
        if len(classif_args) > 0:
            self.classifier.set_params(**classif_args)
        if len(trafo_args) > 0:
            self._set_params("_transformers", **trafo_args)
        return self

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # imports
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
        from sktime.transformations.series.exponent import ExponentTransformer

        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        c = KNeighborsTimeSeriesClassifier()

        # construct without names
        params = {"transformers": [t1, t2], "classifier": c}

        return params
