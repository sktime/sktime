#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements classifier for selecting among different model classes."""
# based on MultiplexForecaster

from sktime.base import _HeterogenousMetaEstimator
from sktime.classification._delegate import _DelegatedClassifier
from sktime.classification.base import BaseClassifier
from sktime.datatypes import MTYPE_LIST_PANEL, MTYPE_LIST_TABLE

__author__ = ["fkiraly"]
__all__ = ["MultiplexClassifier"]


class MultiplexClassifier(_HeterogenousMetaEstimator, _DelegatedClassifier):
    """MultiplexClassifier for selecting among different models.

    MultiplexClassifier facilitates a framework for performing
    model selection process over different model classes.
    It should be used in conjunction with GridSearchCV to get full utilization.
    It can be used with univariate and multivariate classifiers,
    single-output and multi-output classifiers.

    MultiplexClassifier is specified with a (named) list of classifiers
    and a selected_classifier hyper-parameter, which is one of the classifier names.
    The MultiplexClassifier then behaves precisely as the classifier with
    name selected_classifier, ignoring functionality in the other classifiers.

    When used with GridSearchCV, MultiplexClassifier
    provides an ability to tune across multiple estimators, i.e., to perform AutoML,
    by tuning the selected_classifier hyper-parameter. This combination will then
    select one of the passed classifiers via the tuning algorithm.

    Parameters
    ----------
    classifiers : list of sktime classifiers, or
        list of tuples (str, estimator) of sktime classifiers
        MultiplexClassifier can switch ("multiplex") between these classifiers.
        These are "blueprint" classifiers, states do not change when ``fit`` is called.
    selected_classifier: str or None, optional, Default=None.
        If str, must be one of the classifier names.
            If no names are provided, must coincide with auto-generated name strings.
            To inspect auto-generated name strings, call get_params.
        If None, behaves as if the first classifier in the list is selected.
        Selects the classifier as which MultiplexClassifier behaves.

    Attributes
    ----------
    classifier_ : sktime classifier
        clone of the selected classifier used for fitting and classification.
    _classifiers : list of (str, classifier) tuples
        str are identical to those passed, if passed strings are unique
        otherwise unique strings are generated from class name; if not unique,
        the string ``_[i]`` is appended where ``[i]`` is count of occurrence up until
        then
    """

    _tags = {
        "authors": ["fkiraly"],
        "capability:multioutput": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": MTYPE_LIST_PANEL,
        "y_inner_mtype": MTYPE_LIST_TABLE,
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # attribute for _DelegatedClassifier, which then delegates
    #     all non-overridden methods to those of same name in self.classifier_
    #     see further details in _DelegatedClassifier docstring
    _delegate_name = "classifier_"

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_classifiers"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "classifiers_"

    def __init__(
        self,
        classifiers: list,
        selected_classifier=None,
    ):
        super().__init__()
        self.selected_classifier = selected_classifier

        self.classifiers = classifiers
        self._check_estimators(
            classifiers,
            attr_name="classifiers",
            cls_type=BaseClassifier,
            clone_ests=False,
        )
        self._set_classifier()

        self.clone_tags(self.classifier_)
        self.set_tags(**{"fit_is_empty": False})
        # this ensures that we convert in the inner estimator, not in the multiplexer
        self.set_tags(**{"X_inner_mtype": MTYPE_LIST_PANEL})
        self.set_tags(**{"y_inner_mtype": MTYPE_LIST_TABLE})

    @property
    def _classifiers(self):
        """Classifiers turned into name/est tuples."""
        return self._get_estimator_tuples(self.classifiers, clone_ests=False)

    @_classifiers.setter
    def _classifiers(self, value):
        self.classifiers = value

    def _check_selected_classifier(self):
        component_names = self._get_estimator_names(self._classifiers, make_unique=True)
        selected = self.selected_classifier
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_classifier parameter value provided, "
                f" found: {self.selected_classifier}. Must be one of these"
                f" valid selected_classifier parameter values: {component_names}."
            )

    def __or__(self, other):
        """Magic | (or) method, return (right) concatenated MultiplexClassifier.

        Implemented for ``other`` being a classifier, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` classifier, must inherit from BaseClassifier
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        MultiplexClassifier object, concatenation of ``self`` (first) with ``other``
        (last).
            not nested, contains only non-MultiplexClassifier ``sktime`` classifiers

        Raises
        ------
        ValueError if other is not of type MultiplexClassifier or BaseClassifier.
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseClassifier,
            composite_class=MultiplexClassifier,
            attr_name="classifiers",
            concat_order="left",
        )

    def __ror__(self, other):
        """Magic | (or) method, return (left) concatenated MultiplexClassifier.

        Implemented for ``other`` being a classifier, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` classifier, must inherit from BaseClassifier
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        MultiplexClassifier object, concatenation of ``self`` (last) with ``other``
        (first).
            not nested, contains only non-MultiplexClassifier ``sktime`` classifiers
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseClassifier,
            composite_class=MultiplexClassifier,
            attr_name="classifiers",
            concat_order="right",
        )

    def _set_classifier(self):
        self._check_selected_classifier()
        # clone the selected classifier to self.classifier_
        if self.selected_classifier is not None:
            for name, classifier in self._get_estimator_tuples(self.classifiers):
                if self.selected_classifier == name:
                    self.classifier_ = classifier.clone()
        else:
            # if None, simply clone the first classifier to self.classifier_
            self.classifier_ = self._get_estimator_list(self.classifiers)[0].clone()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.classification.dummy import DummyClassifier

        params1 = {
            "classifiers": [
                ("Naive_maj", DummyClassifier(strategy="most_frequent")),
                ("Naive_pri", DummyClassifier(strategy="prior")),
                ("Naive_uni", DummyClassifier(strategy="uniform")),
            ],
            "selected_classifier": "Naive_maj",
        }
        params2 = {
            "classifiers": [
                DummyClassifier(strategy="most_frequent"),
                DummyClassifier(strategy="prior"),
                DummyClassifier(strategy="uniform"),
            ],
        }
        return [params1, params2]
