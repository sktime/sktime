"""Using a clusterer as transformation."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from sktime.datatypes import MTYPE_LIST_PANEL
from sktime.transformations.base import BaseTransformer

__author__ = ["fkiraly"]
__all__ = ["ClustererAsTransformer"]


class ClustererAsTransformer(BaseTransformer):
    """Use a clusterer as a transformer.

    This adapter is used in coercions, when passing a clusterer to a transformer slot.

    The transformation is series-to-primitives, transforming a time series
    into its cluster assignment.

    The adapter dispatches ``BaseTransformer.transform`` to ``BaseClusterer.predict``,
    and requires a clusterer that is able to make cluster assignments via ``predict``,
    see the ``capability:predict`` tag for clusterers.

    Parameters
    ----------
    clusterer : sktime clusterer, i.e., estimator inheriting from BaseClusterer
        this is a "blueprint" clusterer, state does not change when ``fit`` is called

    Attributes
    ----------
    clusterer_ : sktime clusterer, clone of clusterer in `clusterer`
        this clone is fitted in the pipeline when `fit` is called

    Examples
    --------
    >>> from sktime.clustering.dbscan import TimeSeriesDBSCAN
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.clustering.compose import ClustererAsTransformer
    >>> X, _ = load_unit_test(split="train")
    >>> cluster_assign_trafo = ClustererAsTransformer(TimeSeriesDBSCAN())
    >>> cluster_assign_trafo.fit(X)
    ClustererAsTransformer(...)
    >>> cluster_assignment = cluster_assign_trafo.transform(X)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": MTYPE_LIST_PANEL,
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "handles-missing-data": True,
        "capability:missing_values:removes": True,
    }

    def __init__(self, clusterer):
        self.clusterer = clusterer
        self.clusterer_ = clusterer.clone()
        super().__init__()

        can_transform = clusterer.get_tag("capability:predict", False)
        if not can_transform:
            raise ValueError(
                f"To use clusterer {type(clusterer).__name__} as a transformation, "
                "it must support cluster assignment via predict, but it does not. "
                'See the tag "capability:predict" for cluster assignment capability.'
            )

        multivariate = clusterer.get_tag("capability:multivariate", False)
        missing = clusterer.get_tag("capability:missing_values", False)
        unequal = clusterer.get_tag("capability:unequal_length", False)

        # forward tag information
        tags_to_set = {
            "univariate-only": not multivariate,
            "handles-missing-data": missing,
            "capability:unequal_length": unequal,
        }
        self.set_tags(**tags_to_set)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("univariate-only")==True:
                guaranteed to have a single column
            if self.get_tag("univariate-only")==False: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        self: reference to self
        """
        self.clusterer_.fit(X)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("univariate-only")==True:
                guaranteed to have a single column
            if self.get_tag("univariate-only")==False: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        transformed version of X
        """
        return self.clusterer_.predict(X)

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
        from sktime.clustering.k_means import TimeSeriesKMeansTslearn
        from sktime.utils.dependencies import _check_estimator_deps

        params = []

        # construct without names
        c = TimeSeriesDBSCAN.create_test_instance()

        params1 = {"clusterer": c}
        params = params + [params1]

        if _check_estimator_deps(TimeSeriesKMeansTslearn, severity="none"):
            c = TimeSeriesKMeansTslearn.create_test_instance()

            params2 = {"clusterer": c}
        else:
            tsd_params2 = TimeSeriesDBSCAN.get_test_params()[1]
            params2 = {"clusterer": TimeSeriesDBSCAN(**tsd_params2)}

        params = params + [params2]

        return params
