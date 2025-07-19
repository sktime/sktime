"""Implements ParamFitterPipeline.

A class to create a pipeline of transformers and a parameter estimator.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from sktime.base import _HeterogenousMetaEstimator
from sktime.param_est.base import BaseParamFitter
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline

__author__ = ["fkiraly"]
__all__ = ["ParamFitterPipeline"]

# we ensure that internally we convert to pandas for now
SUPPORTED_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


class ParamFitterPipeline(_HeterogenousMetaEstimator, BaseParamFitter):
    """Pipeline of transformers and a parameter estimator.

    The ``ParamFitterPipeline`` compositor chains transformers and a single estimator.
    The pipeline is constructed with a list of sktime transformers, plus an estimator,
        i.e., estimators following the BaseTransformer, ParamFitterPipeline interfaces.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers ``trafo1``, ``trafo2``, ..., ``trafoN`` and an estimator
    ``est``,
        the pipeline behaves as follows:
    ``fit(X)`` - changes state by running ``trafo1.fit_transform`` on ``X``,
        them ``trafo2.fit_transform`` on the output of ``trafo1.fit_transform``, etc
        sequentially, with ``trafo[i]`` receiving the output of ``trafo[i-1]``,
        and then running ``est.fit`` with ``X`` being the output of ``trafo[N]``
    ``update(X)`` - changes state by running ``trafo1.update.transform`` on ``X``,
        them ``trafo2.update.transform`` on the output of ``trafo1.update.transform``,
        etc
        sequentially, with ``trafo[i]`` receiving the output of ``trafo[i-1]``,
        and then running ``est.update`` with ``X`` being the output of ``trafo[N]``

    ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, ``f"_{str(i)}"`` is appended to each name string
            where ``i`` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    ``ParamFitterPipeline`` can also be created by using the magic multiplication
        on any parameter estimator, i.e., if ``est`` inherits from ``BaseParamFitter``,
            and ``my_trafo1``, ``my_trafo2`` inherit from ``BaseTransformer``, then,
            for instance, ``my_trafo1 * my_trafo2 * est``
            will result in the same object as  obtained from the constructor
            ``ParamFitterPipeline(param_est=est, transformers=[my_trafo1, my_trafo2])``
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    param_est : parameter estimator, i.e., estimator inheriting from BaseParamFitter
        this is a "blueprint" estimator, state does not change when ``fit`` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when ``fit`` is called

    Attributes
    ----------
    param_est_ : sktime estimator, clone of estimator in ``param_est``
        this clone is fitted in the pipeline when ``fit`` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in ``transformers`` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in ``transformers_`` is clone of i-th in ``transformers``

    Examples
    --------
    >>> from sktime.param_est.compose import ParamFitterPipeline
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.series.difference import Differencer
    >>> from sktime.datasets import load_airline
    >>>
    >>> X = load_airline()
    >>> pipe = ParamFitterPipeline(SeasonalityACF(), [Differencer()])  # doctest: +SKIP
    >>> pipe.fit(X)  # doctest: +SKIP
    ParamFitterPipeline(...)
    >>> pipe.get_fitted_params()["sp"]  # doctest: +SKIP
    12

    Alternative construction via dunder method:

    >>> pipe = Differencer() * SeasonalityACF()  # doctest: +SKIP
    """

    _tags = {
        "authors": "fkiraly",
        "X_inner_mtype": SUPPORTED_MTYPES,
        # which types do _fit/_predict, support for X?
        "scitype:X": ["Series", "Panel", "Hierarchical"],
        # which X scitypes are supported natively?
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
    }

    # no default tag values - these are set dynamically below

    def __init__(self, param_est, transformers):
        self.param_est = param_est
        self.param_est_ = param_est.clone()
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff: both estimator and all transformers can
        multivariate = param_est.get_tag("capability:multivariate", False)
        multivariate = multivariate and not self.transformers_.get_tag(
            "univariate-only", True
        )
        # can handle missing values iff: both estimator and all transformers can,
        #   *or* transformer chain removes missing data
        missing = param_est.get_tag("capability:missing_values", False)
        missing = missing and self.transformers_.get_tag(
            "capability:missing_values", False
        )
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )

        # set the pipeline tags to the above
        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
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
            self._coerce_estimator_tuple(self.param_est)
        ]

    @property
    def steps_(self):
        return self._transformers + [self._coerce_estimator_tuple(self.param_est_)]

    def __rmul__(self, other):
        """Magic * method, return concatenated ParamFitterPipeline, trafos on left.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        ParamFitterPipeline object, concatenation of ``other`` (first) with ``self``
        (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a ParamFitterPipeline
            new_pipeline = ParamFitterPipeline(
                param_est=self.param_est,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        Xt = self.transformers_.fit_transform(X)
        self.param_est_.fit(Xt)
        return self

    def _update(self, X):
        """Update fitted parameters on more data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Writes to self:
            Sets fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series with which to update the estimator.

        Returns
        -------
        self : reference to self
        """
        Xt = self.transformers_.update(X).transform(X)
        self.param_est_.update(Xt)
        return self

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        return self.param_est_.get_fitted_params()

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
        if "param_est" in kwargs.keys():
            if not isinstance(kwargs["param_est"], BaseParamFitter):
                raise TypeError('"param_est" arg must be an sktime parameter fitter')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        parest_keys = self.param_est.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        parest_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=parest_keys)
        if len(parest_args) > 0:
            self.param_est.set_params(**parest_args)
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
        from sktime.param_est.fixed import FixedParams
        from sktime.param_est.seasonality import SeasonalityACF
        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.utils.dependencies import _check_estimator_deps

        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        p0 = FixedParams({"a": 2, 3: 42})

        # construct with name/estimator tuples
        params = [{"transformers": [("foo", t1), ("bar", t2)], "param_est": p0}]

        # test case 2 depends on statsmodels, requires statsmodels
        if _check_estimator_deps(SeasonalityACF, severity="none"):
            p = SeasonalityACF()

            # construct without names
            params = params + [{"transformers": [t1, t2], "param_est": p}]

        return params
