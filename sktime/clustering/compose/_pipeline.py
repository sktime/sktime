"""Pipeline with a clusterer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
import numpy as np

from sktime.base import _HeterogenousMetaEstimator
from sktime.clustering.base import BaseClusterer
from sktime.datatypes import convert_to
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline
from sktime.utils.sklearn import is_sklearn_clusterer

__author__ = ["fkiraly"]
__all__ = ["ClustererPipeline", "SklearnClustererPipeline"]


class ClustererPipeline(_HeterogenousMetaEstimator, BaseClusterer):
    """Pipeline of transformers and a clusterer.

    The `ClustererPipeline` compositor chains transformers and a single clusterer.
    The pipeline is constructed with a list of sktime transformers, plus a clusterer,
        i.e., estimators following the BaseTransformer resp BaseClusterer interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a clusterer `clst`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `clst.fit` with `X` being the output of `trafo[N]`,
        and `y` identical with the input to `self.fit`
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clst.predict` on the output of `trafoN.transform`,
        and returning the output of `clst.predict`
    `predict_proba(X)` - result is of executing `trafo1.transform`, `trafo2.transform`,
        etc, with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clst.predict_proba` on the output of `trafoN.transform`,
        and returning the output of `clst.predict_proba`

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `ClustererPipeline` can also be created by using the magic multiplication
        on any clusterer, i.e., if `my_clst` inherits from `BaseClusterer`,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_clst`
            will result in the same object as  obtained from the constructor
            `ClustererPipeline(clusterer=my_clst, transformers=[my_trafo1, my_trafo2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    clusterer : sktime clusterer, i.e., estimator inheriting from BaseClusterer
        this is a "blueprint" clusterer, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    clusterer_ : sktime clusterer, clone of clusterer in `clusterer`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sktime.transformations.panel.pca import PCATransformer
    >>> from sktime.clustering.k_means import TimeSeriesKMeans
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.clustering.compose import ClustererPipeline
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
    >>> pipeline = ClustererPipeline(
    ...     TimeSeriesKMeans(), [PCATransformer()]
    ... ) # doctest: +SKIP
    >>> pipeline.fit(X_train, y_train) # doctest: +SKIP
    ClustererPipeline(...)
    >>> y_pred = pipeline.predict(X_test) # doctest: +SKIP

    Alternative construction via dunder method:

    >>> pipeline = PCATransformer() * TimeSeriesKMeans() # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "fkiraly",
        # estimator type
        # --------------
        "X_inner_mtype": "pd-multiindex",  # which type do _fit/_predict accept
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    # no default tag values - these are set dynamically below

    def __init__(self, clusterer, transformers):
        self.clusterer = clusterer
        self.clusterer_ = clusterer.clone()
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff: both clusterer and all transformers can
        multivariate = clusterer.get_tag("capability:multivariate", False)
        multivariate = multivariate and not self.transformers_.get_tag(
            "univariate-only", True
        )
        # can handle missing values iff: both clusterer and all transformers can,
        #   *or* transformer chain removes missing data
        missing = clusterer.get_tag("capability:missing_values", False)
        missing = missing and self.transformers_.get_tag("handles-missing-data", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff: clusterer can and transformers can,
        #   *or* transformer chain renders the series equal length
        unequal = clusterer.get_tag("capability:unequal_length")
        unequal = unequal and self.transformers_.get_tag(
            "capability:unequal_length", False
        )
        unequal = unequal or self.transformers_.get_tag(
            "capability:unequal_length:removes", False
        )
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

        tags_to_clone = [
            "capability:out_of_sample",
            "capability:predict",
            "capability:predict_proba",
        ]
        self.clone_tags(clusterer, tags_to_clone)

    @property
    def _transformers(self):
        return self.transformers_._steps

    @_transformers.setter
    def _transformers(self, value):
        self.transformers_._steps = value

    @property
    def _steps(self):
        return self._check_estimators(self.transformers) + [
            self._coerce_estimator_tuple(self.clusterer)
        ]

    @property
    def steps_(self):
        return self._transformers + [self._coerce_estimator_tuple(self.clusterer_)]

    def __rmul__(self, other):
        """Magic * method, return concatenated ClustererPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClustererPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a ClustererPipeline
            new_pipeline = ClustererPipeline(
                clusterer=self.clusterer,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        core logic

        Parameters
        ----------
        X : Training data of type self.get_tag("X_inner_mtype")
        y: ignored, present for API consistency

        Returns
        -------
        self : reference to self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        Xt = self.transformers_.fit_transform(X=X, y=y)
        self.clusterer_.fit(X=Xt, y=y)

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
        return self.clusterer_.predict(X=Xt)

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
        return self.clusterer_.predict_proba(Xt)

    def _score(self, X, y=None):
        """Score the clustering result."""
        Xt = self.transformers_.transform(X=X)
        return self.clusterer_.score(X=Xt)

    def get_params(self, deep=True):
        """Get parameters of estimator in `transformers`.

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
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "clusterer" in kwargs.keys():
            if not isinstance(kwargs["clusterer"], BaseClusterer):
                raise TypeError('"clusterer" arg must be an sktime clusterer')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        classif_keys = self.clusterer.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        classif_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=classif_keys)
        if len(classif_args) > 0:
            self.clusterer.set_params(**classif_args)
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
            special parameters are defined for a value, will return `"default"` set.
            For clusterers, a "default" set of parameters should be provided for
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
        # imports
        from sktime.clustering.dbscan import TimeSeriesDBSCAN
        from sktime.clustering.k_means import TimeSeriesKMeans
        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.utils.dependencies import _check_estimator_deps

        params = []

        # construct without names
        t1 = ExponentTransformer(power=2)
        c = TimeSeriesDBSCAN.create_test_instance()

        params1 = {"transformers": [t1], "clusterer": c}
        params = params + [params1]

        if _check_estimator_deps(TimeSeriesKMeans, severity="none"):
            t1 = ExponentTransformer(power=2)
            t2 = ExponentTransformer(power=0.5)
            c = TimeSeriesKMeans(random_state=42)

            params2 = {"transformers": [t1, t2], "clusterer": c}

            params = params + [params2]

        return params


class SklearnClustererPipeline(ClustererPipeline):
    """Pipeline of transformers and a clusterer.

    The `SklearnClustererPipeline` chains transformers and an single clusterer.
        Similar to `ClustererPipeline`, but uses a tabular `sklearn` clusterer.
    The pipeline is constructed with a list of sktime transformers, plus a clusterer,
        i.e., transformers following the BaseTransformer interface,
        clusterer follows the `scikit-learn` clusterer interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a clusterer `clst`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`, and
        then running `clst.fit` with `X` the output of `trafo[N]` converted to numpy,
        and `y` identical with the input to `self.fit`.
        `X` is converted to `numpyflat` mtype if `X` is of `Panel` scitype;
        `X` is converted to `numpy2D` mtype if `X` is of `Table` scitype.
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clst.predict` on the numpy converted output of `trafoN.transform`,
        and returning the output of `clst.predict`.
        Output of `trasfoN.transform` is converted to numpy, as in `fit`.
    `predict_proba(X)` - result is of executing `trafo1.transform`, `trafo2.transform`,
        etc, with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clst.predict_proba` on the output of `trafoN.transform`,
        and returning the output of `clst.predict_proba`.
        Output of `trasfoN.transform` is converted to numpy, as in `fit`.

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `SklearnClustererPipeline` can also be created by using the magic multiplication
        between `sktime` transformers and `sklearn` clusterers,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_clst`
            will result in the same object as  obtained from the constructor
            `SklearnClustererPipeline(clusterer=my_clst, transformers=[t1, t2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    clusterer : sklearn clusterer, i.e., inheriting from sklearn ClustererMixin
        this is a "blueprint" clusterer, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    clusterer_ : sklearn clusterer, clone of clusterer in `clusterer`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.transformations.series.summarize import SummaryTransformer
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.clustering.compose import SklearnClustererPipeline
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> t1 = ExponentTransformer()
    >>> t2 = SummaryTransformer()
    >>> pipeline = SklearnClustererPipeline(KMeans(), [t1, t2])
    >>> pipeline = pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)

    Alternative construction via dunder method:

    >>> pipeline = t1 * t2 * KMeans()
    """

    _tags = {
        "X_inner_mtype": "pd-multiindex",  # which type do _fit/_predict accept
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": True,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    # no default tag values - these are set dynamically below

    def __init__(self, clusterer, transformers):
        from sklearn.base import clone

        self.clusterer = clusterer
        self.clusterer_ = clone(clusterer)
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super(ClustererPipeline, self).__init__()

        # can handle multivariate iff all transformers can
        # sklearn transformers always support multivariate
        multivariate = not self.transformers_.get_tag("univariate-only", True)
        # can handle missing values iff transformer chain removes missing data
        # sklearn clusterers might be able to handle missing data (but no tag there)
        # so better set the tag liberally
        missing = self.transformers_.get_tag("handles-missing-data", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff transformer chain renders series equal length
        # because sklearn clusterers require equal length (number of variables) input
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
            self._coerce_estimator_tuple(self.clusterer)
        ]

    @property
    def steps_(self):
        return self._transformers + [self._coerce_estimator_tuple(self.clusterer_)]

    def __rmul__(self, other):
        """Magic * method, return concatenated ClustererPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClustererPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a SklearnClustererPipeline
            new_pipeline = SklearnClustererPipeline(
                clusterer=self.clusterer,
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
                f"unexpected X output type in {type(self.clusterer).__name__}, "
                f'in tag "scitype:transform-output", found "{X_scitype}", '
                'expected one of "Primitives" or "Series"'
            )

        return Xt

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        core logic

        Parameters
        ----------
        X : Training data of type self.get_tag("X_inner_mtype")
        y: ignored, present for API consistency

        Returns
        -------
        self : reference to self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        Xt = self.transformers_.fit_transform(X=X, y=y)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        self.clusterer_.fit(Xt_sklearn, y)

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
        return self.clusterer_.predict(Xt_sklearn)

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
        if hasattr(self.clusterer_, "predict_proba"):
            Xt_sklearn = self._convert_X_to_sklearn(Xt)
            return self.clusterer_.predict_proba(Xt_sklearn)
        else:
            # if sklearn clusterer does not have predict_proba
            return BaseClusterer._predict_proba(self, X)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "clusterer" in kwargs.keys():
            if not is_sklearn_clusterer(kwargs["clusterer"]):
                raise TypeError('"clusterer" arg must be an sklearn clusterer')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        classif_keys = self.clusterer.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        classif_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=classif_keys)
        if len(classif_args) > 0:
            self.clusterer.set_params(**classif_args)
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
            special parameters are defined for a value, will return `"default"` set.
            For clusterers, a "default" set of parameters should be provided for
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
        from sklearn.cluster import KMeans

        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.transformations.series.summarize import SummaryTransformer

        # example with series-to-series transformer before sklearn clusterer
        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        c = KMeans(random_state=42, n_init=10)
        params1 = {"transformers": [t1, t2], "clusterer": c}

        # example with series-to-primitive transformer before sklearn clusterer
        t1 = ExponentTransformer(power=2)
        t2 = SummaryTransformer()
        c = KMeans(random_state=42, n_init=10)
        params2 = {"transformers": [t1, t2], "clusterer": c}

        # construct without names
        return [params1, params2]
