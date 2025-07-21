"""Using an detection estimator as transformation."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from sktime.datatypes import MTYPE_LIST_SERIES
from sktime.transformations.base import BaseTransformer

__author__ = ["fkiraly"]
__all__ = ["AnnotatorAsTransformer", "DetectorAsTransformer"]


MTYPE_LIST_FOR_DETECTORS = MTYPE_LIST_SERIES
# override until annotators only support pd.Series
MTYPE_LIST_FOR_DETECTORS = ["pd.Series"]


class DetectorAsTransformer(BaseTransformer):
    """Use an anomaly, changepoint detector, segmentation estimator as a transformer.

    This adapter is used in coercions, when passing an detector to a transformer slot.

    The transformation is series-to-primitives, transforming a time series
    into its cluster assignment.

    The adapter dispatches ``BaseTransformer.transform`` to
    ``BaseDetector.transform``.

    Parameters
    ----------
    estimator : sktime detector, i.e., estimator inheriting from BaseDetector
        this is a "blueprint" clusterer, state does not change when ``fit`` is called

    Attributes
    ----------
    estimator_ : sktime detector, clone of ``estimator``
        this clone is fitted in the pipeline when ``fit`` is called

    Examples
    --------
    >>> from sktime.detection.compose import DetectorAsTransformer
    >>> from sktime.detection.lof import SubLOF
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical()
    >>> detector = SubLOF.create_test_instance()
    >>> t = DetectorAsTransformer(detector)
    >>> t.fit(X)
    DetectorAsTransformer(...)
    >>> Xt = t.transform(X)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": MTYPE_LIST_FOR_DETECTORS,
        "y_inner_mtype": MTYPE_LIST_FOR_DETECTORS,
        "requires_y": False,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "capability:missing_values": True,
        "capability:missing_values:removes": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_ = estimator.clone()
        super().__init__()

        requires_y = estimator.get_tag("learning_type", "unsupervised") == "supervised"

        # forward tag information
        tags_to_set = {"requires_y": requires_y}
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
        self.estimator_.fit(X=X, y=y)
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
        return self.estimator_.transform(X)

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
        from sktime.detection.lof import SubLOF

        params1 = {"estimator": SubLOF.create_test_instance()}
        params2 = {"estimator": SubLOF.create_test_instance()}

        return [params1, params2]


# todo 1.0.0 - remove alias, i.e., remove this line
AnnotatorAsTransformer = DetectorAsTransformer
