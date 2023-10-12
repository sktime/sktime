"""Pipelines for pairwise panel transformers."""

__author__ = ["fkiraly"]

from sktime.base import _HeterogenousMetaEstimator
from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


class PwTrafoPanelPipeline(_HeterogenousMetaEstimator, BasePairwiseTransformerPanel):
    """Pipeline of transformers and a pairwise panel transformer.

    `PwTrafoPanelPipeline` chains transformers and a pairwise transformer at the end.
    The pipeline is constructed with a list of sktime transformers (BaseTransformer),
        plus a pairwise panel transformer, following BasePairwiseTransformerPanel.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and an estimator `est`,
        the pipeline behaves as follows:
    `transform(X)` - running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`.
        Then passes output of `trafo[N]` to `pw_trafo.transform`, as `X`.
        Same chain of transformers is run on `X2` and passed, if not `None`.

    `PwTrafoPanelPipeline` can also be created by using the magic multiplication
        on any parameter estimator: if `pw_t` is `BasePairwiseTransformerPanel`,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * pw_t`
            will result in the same object as  obtained from the constructor
            `PwTrafoPanelPipeline(pw_trafo=pw_t, transformers=[my_trafo1, my_trafo2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    pw_trafo : pairwise panel transformer,
        i.e., estimator inheriting from BasePairwiseTransformerPanel
        this is a "blueprint" estimator, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Examples
    --------
    >>> from sktime.dists_kernels.compose import PwTrafoPanelPipeline
    >>> from sktime.dists_kernels.dtw import DtwDist
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.datasets import load_unit_test
    >>>
    >>> X, _ = load_unit_test()
    >>> X = X[0:3]
    >>> pipeline = PwTrafoPanelPipeline(DtwDist(), [ExponentTransformer()])
    >>> dist_mat = pipeline.transform(X)
    """

    _tags = {
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
    }

    def __init__(self, pw_trafo, transformers):
        self.pw_trafo = pw_trafo
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff: both classifier and all transformers can
        multivariate = pw_trafo.get_tag("capability:multivariate", False)
        multivariate = multivariate and not self.transformers_.get_tag(
            "univariate-only", True
        )
        # can handle missing values iff: both classifier and all transformers can,
        #   *or* transformer chain removes missing data
        missing = pw_trafo.get_tag("capability:missing_values", False)
        missing = missing and self.transformers_.get_tag("handles-missing-data", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )

        # type of trafo is the same
        trafo_type = pw_trafo.get_tag("pwtrafo_type", "distance")

        # set the pipeline tags to the above
        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "pwtrafo_type": trafo_type,
        }
        self.set_tags(**tags_to_set)

    @property
    def _transformers(self):
        return self.transformers_._steps

    @_transformers.setter
    def _transformers(self, value):
        self.transformers_._steps = value

    def __rmul__(self, other):
        """Magic * method, return concatenated PwTrafoPanelPipeline, trafos on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ParamFitterPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a PwTrafoPanelPipeline
            new_pipeline = PwTrafoPanelPipeline(
                pw_trafo=self.pw_trafo,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from public transform

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: sktime Panel data container
        X2: sktime Panel data container

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        trafos = self.transformers_.clone()
        pw_trafo = self.pw_trafo

        Xt = trafos.fit_transform(X)

        # find out whether we know that the resulting matrix is symmetric
        #   since aligner distances are always symmetric,
        #   we know it's the case for sure if X equals X2
        if X2 is None:
            X2t = None
        else:
            X2t = trafos.fit_transform(X2)

        distmat = pw_trafo.transform(Xt, X2t)
        return distmat

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
        if "pw_trafo" in kwargs.keys():
            if not isinstance(kwargs["pw_trafo"], BasePairwiseTransformerPanel):
                raise TypeError('"pw_trafo" arg must be a pairwise panel transformer')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        pw_keys = self.pw_trafo.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        pw_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=pw_keys)
        if len(pw_args) > 0:
            self.pw_trafo.set_params(**pw_args)
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
            There are currently no reserved values for distance/kernel transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.dists_kernels.compose_tab_to_panel import AggrDist, FlatDist
        from sktime.transformations.series.boxcox import BoxCoxTransformer
        from sktime.transformations.series.exponent import ExponentTransformer

        # transformer has no fit, two transformers, list without names
        params1 = {
            "pw_trafo": AggrDist.create_test_instance(),
            "transformers": [ExponentTransformer(), ExponentTransformer(3)],
        }

        # transformer has fit, 1 transformer, name/estimator pair list
        params2 = {
            "pw_trafo": FlatDist.create_test_instance(),
            "transformers": [("boxcox", BoxCoxTransformer())],
        }

        return [params1, params2]
