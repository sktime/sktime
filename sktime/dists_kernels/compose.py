# -*- coding: utf-8 -*-
"""Pipelines for pairwise panel transformers."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.base import _HeterogenousMetaEstimator
from sktime.dists_kernels._base import BasePairwiseTransformerPanel
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline


SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


class PwTrafoPanelPipeline(BasePairwiseTransformerPanel, _HeterogenousMetaEstimator):
    """Pipeline of transformers and a parameter estimator.

    The `ParamFitterPipeline` compositor chains transformers and a single estimator.
    The pipeline is constructed with a list of sktime transformers, plus an estimator,
        i.e., estimators following the BaseTransformer, ParamFitterPipeline interfaces.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and an estimator `est`,
        the pipeline behaves as follows:
    `fit(X)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `est.fit` with `X` being the output of `trafo[N]`
    `update(X)` - changes styte by running `trafo1.update.transform` on `X`,
        them `trafo2.update.transform` on the output of `trafo1.update.transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `est.update` with `X` being the output of `trafo[N]`

    `ParamFitterPipeline` can also be created by using the magic multiplication
        on any parameter estimator, i.e., if `est` inherits from `BaseParamFitter`,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * est`
            will result in the same object as  obtained from the constructor
            `ParamFitterPipeline(classifier=est, transformers=[my_trafo1, my_trafo2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    pw_trafo : pairwise panel transformer,
        i.e., estimator inheriting from BasePairwiseTransformerPane
        this is a "blueprint" estimator, state does not change when `fit` is called
    transformers : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    pw_trafo_ : sktime estimator, clone of estimator in `pw_trafo`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sktime.param_est.compose import ParamFitterPipeline
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.series.difference import Differencer
    >>> from sktime.datasets import load_airline
    >>>
    >>> X = load_airline()
    >>> pipeline = ParamFitterPipeline(SeasonalityACF(), [Differencer()])
    >>> pipeline.fit(X)
    ParamFitterPipeline(...)
    >>> pipeline.get_fitted_params()["sp"]
    12

    Alternative construction via dunder method:
    >>> pipeline = Differencer() * SeasonalityACF()
    """

    _tags = {
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
    }

    # no default tag values - these are set dynamically below

    def __init__(self, pw_trafo, transformers):

        self.pw_trafo = pw_trafo
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super(PwTrafoPanelPipeline, self).__init__()

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

            Core logic

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        # find out whether we know that the resulting matrix is symmetric
        #   since aligner distances are always symmetric,
        #   we know it's the case for sure if X equals X2
        if X2 is None:
            X = X2
            symm = True
        else:
            symm = False

        n = len(X)
        m = len(X2)

        distmat = np.zeros((n, m), dtype="float")

        if self.aligner is not None:
            aligner = self.aligner.clone()
        else:
            return distmat

        for i in range(n):
            for j in range(m):
                if symm and j < i:
                    distmat[i, j] = distmat[j, i]
                else:
                    distmat[i, j] = aligner.fit([X[i], X2[j]]).get_distance()

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for DistFromAligner."""
        # importing inside to avoid circular dependencies
        from sktime.alignment.dtw_python import AlignerDTW
        from sktime.utils.validation._dependencies import _check_estimator_deps

        if _check_estimator_deps(AlignerDTW, severity="none"):
            return {"aligner": AlignerDTW()}
        else:
            return {}
