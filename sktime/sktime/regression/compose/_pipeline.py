"""Pipeline with a regressor."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
import numpy as np

from sktime.base import _HeterogenousMetaEstimator
from sktime.datatypes import convert_to
from sktime.regression.base import BaseRegressor
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline
from sktime.utils.sklearn import is_sklearn_regressor

__author__ = ["fkiraly"]
__all__ = ["RegressorPipeline", "SklearnRegressorPipeline"]


class RegressorPipeline(_HeterogenousMetaEstimator, BaseRegressor):
    """Pipeline of transformers and a regressor.

    The `RegressorPipeline` compositor chains transformers and a single regressor.
    The pipeline is constructed with a list of sktime transformers, plus a regressor,
        i.e., estimators following the BaseTransformer resp BaseRegressor interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a regressor `reg`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `reg.fit` with `X` being the output of `trafo[N]`,
        and `y` identical with the input to `self.fit`
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `reg.predict` on the output of `trafoN.transform`,
        and returning the output of `reg.predict`

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `RegressorPipeline` can also be created by using the magic multiplication
        on any regressor, i.e., if `my_reg` inherits from `BaseRegressor`,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_reg`
            will result in the same object as  obtained from the constructor
            `RegressorPipeline(regressor=my_reg, transformers=[my_trafo1, my_trafo2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    regressor : sktime regressor, i.e., estimator inheriting from BaseRegressor
        this is a "blueprint" regressor, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    regressor_ : sktime regressor, clone of regressor in `regressor`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sktime.transformations.panel.pca import PCATransformer
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.compose import RegressorPipeline
    >>> from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> pipeline = RegressorPipeline(
    ...     KNeighborsTimeSeriesRegressor(n_neighbors=2), [PCATransformer()]
    ... )
    >>> pipeline.fit(X_train, y_train)
    RegressorPipeline(...)
    >>> y_pred = pipeline.predict(X_test)

    Alternative construction via dunder method:

    >>> pipeline = PCATransformer() * KNeighborsTimeSeriesRegressor(n_neighbors=2)
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

    _required_parameters = ["regressor"]

    # no default tag values - these are set dynamically below

    def __init__(self, regressor, transformers):
        self.regressor = regressor
        self.regressor_ = regressor.clone()
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff: both regressor and all transformers can
        multivariate = regressor.get_tag("capability:multivariate", False)
        multivariate = multivariate and not self.transformers_.get_tag(
            "univariate-only", True
        )
        # can handle missing values iff: both regressor and all transformers can,
        #   *or* transformer chain removes missing data
        missing = regressor.get_tag("capability:missing_values", False)
        missing = missing and self.transformers_.get_tag("handles-missing-data", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff: regressor can and transformers can,
        #   *or* transformer chain renders the series equal length
        unequal = regressor.get_tag("capability:unequal_length")
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

    @property
    def _transformers(self):
        return self.transformers_._steps

    @_transformers.setter
    def _transformers(self, value):
        self.transformers_._steps = value

    def __rmul__(self, other):
        """Magic * method, return concatenated RegressorPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a RegressorPipeline
            new_pipeline = RegressorPipeline(
                regressor=self.regressor,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    def _fit(self, X, y):
        """Fit time series regressor to training data.

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
        self.regressor_.fit(X=Xt, y=y)

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
        return self.regressor_.predict(X=Xt)

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
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "regressor" in kwargs.keys():
            if not isinstance(kwargs["regressor"], BaseRegressor):
                raise TypeError('"regressor" arg must be an sktime regressor')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        regr_keys = self.regressor.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        regr_args = self._subset_dict_keys(
            dict_to_subset=kwargs, keys=regr_keys, prefix="regressor"
        )
        if len(regr_args) > 0:
            self.regressor.set_params(**regr_args)
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
            For regressors, a "default" set of parameters should be provided for
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
        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.utils.dependencies import _check_soft_dependencies

        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)

        r = SklearnRegressorPipeline.create_test_instance()

        params1 = {"transformers": [t1, t2], "regressor": r}

        if _check_soft_dependencies("numba", severity="none"):
            from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

            c = KNeighborsTimeSeriesRegressor()

            # construct without names
            params2 = {"transformers": [t1, t2], "regressor": c}
            return [params1, params2]
        else:
            return params1


class SklearnRegressorPipeline(_HeterogenousMetaEstimator, BaseRegressor):
    """Pipeline of transformers and a regressor.

    The `SklearnRegressorPipeline` chains transformers and an single regressor.
        Similar to `RegressorPipeline`, but uses a tabular `sklearn` regressor.
    The pipeline is constructed with a list of sktime transformers, plus a regressor,
        i.e., transformers following the BaseTransformer interface,
        regressor follows the `scikit-learn` regressor interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a regressor `reg`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `reg.fit` with `X` the output of `trafo[N]` converted to numpy,
        and `y` identical with the input to `self.fit`.
        `X` is converted to `numpyflat` mtype if `X` is of `Panel` scitype;
        `X` is converted to `numpy2D` mtype if `X` is of `Table` scitype.
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `reg.predict` on the numpy converted output of `trafoN.transform`,
        and returning the output of `reg.predict`.
        Output of `trasfoN.transform` is converted to numpy, as in `fit`.

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `SklearnRegressorPipeline` can also be created by using the magic multiplication
        between `sktime` transformers and `sklearn` regressors,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_reg`
            will result in the same object as  obtained from the constructor
            `SklearnRegressorPipeline(regressor=my_reg, transformers=[t1, t2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    regressor : sklearn regressor, i.e., inheriting from sklearn RegressorMixin
        this is a "blueprint" regressor, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    regressor_ : sklearn regressor, clone of regressor in `regressor`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.compose import SklearnRegressorPipeline
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.transformations.series.summarize import SummaryTransformer
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> t1 = ExponentTransformer()
    >>> t2 = SummaryTransformer()
    >>> pipeline = SklearnRegressorPipeline(KNeighborsRegressor(), [t1, t2])
    >>> pipeline = pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)

    Alternative construction via dunder method:

    >>> pipeline = t1 * t2 * KNeighborsRegressor()
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

    _required_parameters = ["regressor"]

    # no default tag values - these are set dynamically below

    def __init__(self, regressor, transformers):
        from sklearn.base import clone

        self.regressor = regressor
        self.regressor_ = clone(regressor)
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff all transformers can
        # sklearn transformers always support multivariate
        multivariate = not self.transformers_.get_tag("univariate-only", True)
        # can handle missing values iff transformer chain removes missing data
        # sklearn regressors might be able to handle missing data (but no tag there)
        # so better set the tag liberally
        missing = self.transformers_.get_tag("handles-missing-data", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff transformer chain renders series equal length
        # because sklearn regressors require equal length (number of variables) input
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

    def __rmul__(self, other):
        """Magic * method, return concatenated RegressorPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a SklearnRegressorPipeline
            new_pipeline = SklearnRegressorPipeline(
                regressor=self.regressor,
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
                f"unexpected X output type in {type(self.regressor).__name__}, "
                f'in tag "scitype:transform-output", found "{X_scitype}", '
                'expected one of "Primitives" or "Series"'
            )

        return Xt

    def _fit(self, X, y):
        """Fit time series regressor to training data.

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
        self.regressor_.fit(Xt_sklearn, y)

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
        return self.regressor_.predict(Xt_sklearn)

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
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "regressor" in kwargs.keys():
            if not is_sklearn_regressor(kwargs["regressor"]):
                raise TypeError('"regressor" arg must be an sklearn regressor')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        regr_keys = self.regressor.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        regr_args = self._subset_dict_keys(
            dict_to_subset=kwargs, keys=regr_keys, prefix="regressor"
        )
        if len(regr_args) > 0:
            self.regressor.set_params(**regr_args)
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
            For regressors, a "default" set of parameters should be provided for
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
        from sklearn.neighbors import KNeighborsRegressor

        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.transformations.series.summarize import SummaryTransformer

        # example with series-to-series transformer before sklearn regressor
        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        c = KNeighborsRegressor()
        params1 = {"transformers": [t1, t2], "regressor": c}

        # example with series-to-primitive transformer before sklearn regressor
        t1 = ExponentTransformer(power=2)
        t2 = SummaryTransformer()
        c = KNeighborsRegressor()
        params2 = {"transformers": [t1, t2], "regressor": c}

        # construct without names
        return [params1, params2]
