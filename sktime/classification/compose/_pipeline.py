"""Pipeline with a classifier."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
import numpy as np

from sktime.base import _HeterogenousMetaEstimator
from sktime.classification.base import BaseClassifier
from sktime.datatypes import convert_to
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline
from sktime.utils.sklearn import is_sklearn_classifier

__author__ = ["fkiraly"]
__all__ = ["ClassifierPipeline", "SklearnClassifierPipeline"]


class ClassifierPipeline(_HeterogenousMetaEstimator, BaseClassifier):
    """Pipeline of transformers and a classifier.

    The ``ClassifierPipeline`` compositor chains transformers and a single classifier.
    The pipeline is constructed with a list of sktime transformers, plus a classifier,
        i.e., estimators following the BaseTransformer resp BaseClassifier interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers ``trafo1``, ``trafo2``, ..., ``trafoN`` and a classifier
    ``clf``,
        the pipeline behaves as follows:
    ``fit(X, y)`` - changes styte by running ``trafo1.fit_transform`` on ``X``,
        them ``trafo2.fit_transform`` on the output of ``trafo1.fit_transform``, etc
        sequentially, with ``trafo[i]`` receiving the output of ``trafo[i-1]``,
        and then running ``clf.fit`` with ``X`` being the output of ``trafo[N]``,
        and ``y`` identical with the input to ``self.fit``
    ``predict(X)`` - result is of executing ``trafo1.transform``, ``trafo2.transform``,
    etc
        with ``trafo[i].transform`` input = output of ``trafo[i-1].transform``,
        then running ``clf.predict`` on the output of ``trafoN.transform``,
        and returning the output of ``clf.predict``
    ``predict_proba(X)`` - result is of executing ``trafo1.transform``,
    ``trafo2.transform``,
        etc, with ``trafo[i].transform`` input = output of ``trafo[i-1].transform``,
        then running ``clf.predict_proba`` on the output of ``trafoN.transform``,
        and returning the output of ``clf.predict_proba``

    ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, ``f"_{str(i)}"`` is appended to each name string
            where ``i`` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    ``ClassifierPipeline`` can also be created by using the magic multiplication
        on any classifier, i.e., if ``my_clf`` inherits from ``BaseClassifier``,
            and ``my_trafo1``, ``my_trafo2`` inherit from ``BaseTransformer``, then,
            for instance, ``my_trafo1 * my_trafo2 * my_clf``
            will result in the same object as  obtained from the constructor
            ``ClassifierPipeline(classifier=my_clf, transformers=[my_trafo1,
            my_trafo2])``
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    classifier : sktime classifier, i.e., estimator inheriting from BaseClassifier
        this is a "blueprint" classifier, state does not change when ``fit`` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when ``fit`` is called

    Attributes
    ----------
    classifier_ : sktime classifier, clone of classifier in ``classifier``
        this clone is fitted in the pipeline when ``fit`` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in ``transformers`` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in ``transformers_`` is clone of i-th in ``transformers``

    Examples
    --------
    >>> from sktime.transformations.panel.pca import PCATransformer
    >>> from sktime.classification.interval_based import TimeSeriesForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.classification.compose import ClassifierPipeline
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> pipeline = ClassifierPipeline(
    ...     TimeSeriesForestClassifier(n_estimators=5), [PCATransformer()]
    ... )
    >>> pipeline.fit(X_train, y_train)
    ClassifierPipeline(...)
    >>> y_pred = pipeline.predict(X_test)

    Alternative construction via dunder method:

    >>> pipeline = PCATransformer() * TimeSeriesForestClassifier(n_estimators=5)
    """

    _tags = {
        "authors": ["fkiraly"],
        "X_inner_mtype": "pd-multiindex",  # which type do _fit/_predict accept
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
        "capability:predict_proba": True,
        "capability:categorical_in_X": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # no default tag values - these are set dynamically below

    def __init__(self, classifier, transformers):
        self.classifier = classifier
        self.classifier_ = classifier.clone()
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff: both classifier and all transformers can
        multivariate = classifier.get_tag("capability:multivariate", False)
        multivariate = multivariate and not self.transformers_.get_tag(
            "univariate-only", True
        )
        # can handle missing values iff: both classifier and all transformers can,
        #   *or* transformer chain removes missing data
        missing = classifier.get_tag("capability:missing_values", False)
        missing = missing and self.transformers_.get_tag(
            "capability:missing_values", False
        )
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff: classifier can and transformers can,
        #   *or* transformer chain renders the series equal length
        unequal = classifier.get_tag("capability:unequal_length")
        unequal = unequal and self.transformers_.get_tag(
            "capability:unequal_length", False
        )
        unequal = unequal or self.transformers_.get_tag(
            "capability:unequal_length:removes", False
        )
        # predict_proba is same as that of classifier
        predict_proba = classifier.get_tag("capability:predict_proba")
        # last three tags are always False, since not supported by transformers
        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
            "capability:contractable": False,
            "capability:train_estimate": False,
            "capability:multithreading": False,
            "capability:predict_proba": predict_proba,
        }
        self.set_tags(**tags_to_set)

    @property
    def _transformers(self):
        return self.transformers_._steps

    @_transformers.setter
    def _transformers(self, value):
        self.transformers_._steps = value

    @property
    def _steps(self):
        return self._check_estimators(self.transformers) + [
            self._coerce_estimator_tuple(self.classifier)
        ]

    @property
    def steps_(self):
        return self._transformers + [self._coerce_estimator_tuple(self.classifier_)]

    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of ``other`` (first) with ``self``
        (last).
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
        Xt = self.transformers_.fit_transform(X=X, y=y)
        self.classifier_.fit(X=Xt, y=y)

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
        Xt = self.transformers_.transform(X=X)
        return self.classifier_.predict(X=Xt)

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
        """Get parameters of estimator in ``transformers``.

        Parameters
        ----------
        deep : boolean, optional, default=True
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
        """Set the parameters of estimator in ``transformers``.

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
        classif_args = self._subset_dict_keys(
            dict_to_subset=kwargs, keys=classif_keys, prefix="classifier"
        )
        if len(classif_args) > 0:
            self.classifier.set_params(**classif_args)
        if len(trafo_args) > 0:
            self._set_params("_transformers", **trafo_args)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        # imports
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
        from sktime.classification.dummy import DummyClassifier
        from sktime.transformations.series.exponent import ExponentTransformer

        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        c = KNeighborsTimeSeriesClassifier()

        another_c = DummyClassifier()

        params1 = {"transformers": [t1, t2], "classifier": c}
        params2 = {"transformers": [t1], "classifier": another_c}

        return [params1, params2]


class SklearnClassifierPipeline(_HeterogenousMetaEstimator, BaseClassifier):
    """Pipeline of transformers and a classifier.

    The ``SklearnClassifierPipeline`` chains transformers and an single classifier.
        Similar to ``ClassifierPipeline``, but uses a tabular ``sklearn`` classifier.
    The pipeline is constructed with a list of sktime transformers, plus a classifier,
        i.e., transformers following the BaseTransformer interface,
        classifier follows the ``scikit-learn`` classifier interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers ``trafo1``, ``trafo2``, ..., ``trafoN`` and a classifier
    ``clf``,
        the pipeline behaves as follows:
    ``fit(X, y)`` - changes styte by running ``trafo1.fit_transform`` on ``X``,
        them ``trafo2.fit_transform`` on the output of ``trafo1.fit_transform``, etc
        sequentially, with ``trafo[i]`` receiving the output of ``trafo[i-1]``,
        and then running ``clf.fit`` with ``X`` the output of ``trafo[N]`` converted to
        numpy,
        and ``y`` identical with the input to ``self.fit``.
        ``X`` is converted to ``numpyflat`` mtype if ``X`` is of ``Panel`` scitype;
        ``X`` is converted to ``numpy2D`` mtype if ``X`` is of ``Table`` scitype.
    ``predict(X)`` - result is of executing ``trafo1.transform``, ``trafo2.transform``,
    etc
        with ``trafo[i].transform`` input = output of ``trafo[i-1].transform``,
        then running ``clf.predict`` on the numpy converted output of
        ``trafoN.transform``,
        and returning the output of ``clf.predict``.
        Output of ``trasfoN.transform`` is converted to numpy, as in ``fit``.
    ``predict_proba(X)`` - result is of executing ``trafo1.transform``,
    ``trafo2.transform``,
        etc, with ``trafo[i].transform`` input = output of ``trafo[i-1].transform``,
        then running ``clf.predict_proba`` on the output of ``trafoN.transform``,
        and returning the output of ``clf.predict_proba``.
        Output of ``trasfoN.transform`` is converted to numpy, as in ``fit``.

    ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, ``f"_{str(i)}"`` is appended to each name string
            where ``i`` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    ``SklearnClassifierPipeline`` can also be created by using the magic multiplication
        between ``sktime`` transformers and ``sklearn`` classifiers,
            and ``my_trafo1``, ``my_trafo2`` inherit from ``BaseTransformer``, then,
            for instance, ``my_trafo1 * my_trafo2 * my_clf``
            will result in the same object as  obtained from the constructor
            ``SklearnClassifierPipeline(classifier=my_clf, transformers=[t1, t2])``
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    classifier : sklearn classifier, i.e., inheriting from sklearn ClassifierMixin
        this is a "blueprint" classifier, state does not change when ``fit`` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when ``fit`` is called

    Attributes
    ----------
    classifier_ : sklearn classifier, clone of classifier in ``classifier``
        this clone is fitted in the pipeline when ``fit`` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in ``transformers`` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in ``transformers_`` is clone of i-th in ``transformers``

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.transformations.series.summarize import SummaryTransformer
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.classification.compose import SklearnClassifierPipeline
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> t1 = ExponentTransformer()
    >>> t2 = SummaryTransformer()
    >>> pipeline = SklearnClassifierPipeline(KNeighborsClassifier(), [t1, t2])
    >>> pipeline = pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)

    Alternative construction via dunder method:

    >>> pipeline = t1 * t2 * KNeighborsClassifier()
    """

    _tags = {
        "X_inner_mtype": "pd-multiindex",  # which type do _fit/_predict accept
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
        "capability:predict_proba": True,
        "capability:categorical_in_X": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # no default tag values - these are set dynamically below

    def __init__(self, classifier, transformers):
        from sklearn.base import clone

        self.classifier = classifier
        self.classifier_ = clone(classifier)
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # all sktime and sklearn transformers always support multivariate
        multivariate = True
        # can handle missing values iff transformer chain removes missing data
        # sklearn classifiers might be able to handle missing data (but no tag there)
        # so better set the tag liberally
        missing = self.transformers_.get_tag("capability:missing_values", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff transformer chain renders series equal length
        # because sklearn classifiers require equal length (number of variables) input
        unequal = self.transformers_.get_tag("capability:unequal_length:removes", False)
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

    @property
    def _steps(self):
        return self._check_estimators(self.transformers) + [
            self._coerce_estimator_tuple(self.classifier)
        ]

    @property
    def steps_(self):
        return self._transformers + [self._coerce_estimator_tuple(self.classifier_)]

    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of ``other`` (first) with ``self``
        (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a SklearnClassifierPipeline
            new_pipeline = SklearnClassifierPipeline(
                classifier=self.classifier,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    def _convert_X_to_sklearn(self, X):
        """Convert a Table or Panel X to 2D numpy required by sklearn."""
        X_scitype = self.transformers_.get_tag("scitype:transform-output")
        # if X_scitype is Primitives, output is Table, convert to 2D numpy array
        if X_scitype == "Primitives":
            Xt = convert_to(X, to_type="numpy2D", as_scitype="Table")
        # if X_scitype is Series, output is Panel, convert to 2D numpy array (numpyflat)
        elif X_scitype == "Series":
            Xt = convert_to(X, to_type="numpyflat", as_scitype="Panel")
        else:
            raise TypeError(
                f"unexpected X output type in {type(self.classifier).__name__}, "
                f'in tag "scitype:transform-output", found "{X_scitype}", '
                'expected one of "Primitives" or "Series"'
            )

        return Xt

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
        Xt = self.transformers_.fit_transform(X=X, y=y)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        self.classifier_.fit(Xt_sklearn, y)

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
        Xt = self.transformers_.transform(X=X)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        return self.classifier_.predict(Xt_sklearn)

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
        if hasattr(self.classifier_, "predict_proba"):
            Xt_sklearn = self._convert_X_to_sklearn(Xt)
            return self.classifier_.predict_proba(Xt_sklearn)
        else:
            # if sklearn classifier does not have predict_proba
            return BaseClassifier._predict_proba(self, X)

    def get_params(self, deep=True):
        """Get parameters of estimator in ``transformers``.

        Parameters
        ----------
        deep : boolean, optional, default=True
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
        """Set the parameters of estimator in ``transformers``.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "classifier" in kwargs.keys():
            if not is_sklearn_classifier(kwargs["classifier"]):
                raise TypeError('"classifier" arg must be an sklearn classifier')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        classif_keys = self.classifier.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        classif_args = self._subset_dict_keys(
            dict_to_subset=kwargs, keys=classif_keys, prefix="classifier"
        )
        if len(classif_args) > 0:
            self.classifier.set_params(**classif_args)
        if len(trafo_args) > 0:
            self._set_params("_transformers", **trafo_args)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sklearn.neighbors import KNeighborsClassifier

        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.transformations.series.summarize import SummaryTransformer

        # example with series-to-series transformer before sklearn classifier
        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        c = KNeighborsClassifier()
        params1 = {"transformers": [t1, t2], "classifier": c}

        # example with series-to-primitive transformer before sklearn classifier
        t1 = ExponentTransformer(power=2)
        t2 = SummaryTransformer()
        c = KNeighborsClassifier()
        params2 = {"transformers": [t1, t2], "classifier": c}

        # construct without names
        return [params1, params2]
