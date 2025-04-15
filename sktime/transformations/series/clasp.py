"""ClaSP (Classification Score Profile) Transformer implementation.

Notes
-----
As described in
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
"""

__author__ = ["ermshaua", "patrickzib"]
__all__ = ["ClaSPTransformer"]

from sktime.transformations.base import BaseTransformer


class ClaSPTransformer(BaseTransformer):
    """ClaSP (Classification Score Profile) Transformer.

    Implementation of the Classification Score Profile of a time series.
    ClaSP hierarchically splits a TS into two parts, where each split point is
    determined by training a binary TS classifier for each possible split point and
    selecting the one with highest accuracy, i.e., the one that is best at identifying
    subsequences to be from either of the partitions.

    Parameters
    ----------
    window_length :       int, default = 10
        size of window for sliding.
    scoring_metric :      string, default = ROC_AUC
        the scoring metric to use in ClaSP - choose from ROC_AUC or F1
    exclusion_radius : int
        Exclusion Radius for change points to be non-trivial matches

    Notes
    -----
    As described in
    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }

    Examples
    --------
    >>> from sktime.transformations.series.clasp import ClaSPTransformer
    >>> from sktime.detection.clasp import find_dominant_window_sizes
    >>> from sktime.datasets import load_electric_devices_segmentation
    >>> X, true_period_size, true_cps = load_electric_devices_segmentation()
    >>> dominant_period_size = find_dominant_window_sizes(X) # doctest: +SKIP
    >>> clasp = ClaSPTransformer(window_length=dominant_period_size) # doctest: +SKIP
    >>> clasp.fit(X) # doctest: +SKIP
    >>> profile = clasp.transform(X) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ermshaua", "patrickzib"],
        "maintainers": ["ermshaua"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": True,
        "fit_is_empty": True,
    }

    def __init__(
        self, window_length=10, scoring_metric="ROC_AUC", exclusion_radius=0.05
    ):
        self.window_length = int(window_length)
        self.scoring_metric = scoring_metric
        self.exclusion_radius = exclusion_radius
        super().__init__()

    def _transform(self, X, y=None):
        """Compute ClaSP.

        Takes as input a single time series dataset and returns the
        Classification Score profile for that single time series.

        Parameters
        ----------
        X : 2D numpy.ndarray
           A single pandas series or a 1d numpy array
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 1D numpy.ndarray
            transformed version of X
            ClaSP of the single time series as output
            with length as (n-window_length+1)
        """
        from sktime.transformations.series._clasp_numba import clasp

        scoring_metric_call = self._check_scoring_metric(self.scoring_metric)

        X = X.flatten()
        Xt, _ = clasp(
            X,
            self.window_length,
            score=scoring_metric_call,
            exclusion_radius=self.exclusion_radius,
        )

        return Xt

    def _check_scoring_metric(self, scoring_metric):
        """Check which scoring metric to use.

        Parameters
        ----------
        scoring_metric : string
            Choose from "ROC_AUC" or "F1"

        Returns
        -------
        scoring_metric_call : a callable, keyed by the ``scoring_metric`` input
            _roc_auc_score, if scoring_metric = "ROC_AUC"
            _binary_f1_score, if scoring_metric = "F1"
        """
        from sktime.transformations.series._clasp_numba import (
            _binary_f1_score,
            _roc_auc_score,
        )

        valid_scores = ("ROC_AUC", "F1")

        if scoring_metric not in valid_scores:
            raise ValueError(f"invalid input, please use one of {valid_scores}")

        if scoring_metric == "ROC_AUC":
            return _roc_auc_score
        elif scoring_metric == "F1":
            return _binary_f1_score

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return {"window_length": 5}
