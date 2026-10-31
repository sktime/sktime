"""FLUSS (Fast Low-cost Unipotent Semantic Segmentation) algorithm.

Implementation of the FLUSS algorithm for unsupervised time series
segmentation, wrapping ``stumpy.fluss``.

Notes
-----
Based on the FLUSS algorithm as described in [1]_.

References
----------
.. [1] Gharghabi, S., Ding, Y., Yeh, C.-C. M., Kamgar, K., Ulanova, L.,
   & Keogh, E. (2017). Matrix Profile VIII: Domain Agnostic Online Semantic
   Segmentation at Superhuman Performance Levels. In *IEEE International
   Conference on Data Mining (ICDM)* (pp. 117-126).
"""

__author__ = ["HimanshuBairwa"]
__all__ = ["FLUSSSegmenter"]

import pandas as pd

from sktime.detection.base import BaseDetector


class FLUSSSegmenter(BaseDetector):
    """FLUSS-based unsupervised change-point detector.

    This implementation wraps ``stumpy.fluss`` to detect regime changes in a
    univariate time series.  A matrix profile is computed via ``stumpy.stump``
    and then segmented with ``stumpy.fluss``.

    Parameters
    ----------
    window_length : int
        Window size (``m``) used for matrix profile computation.
    n_regimes : int
        Number of regimes to detect.  This corresponds to approximately
        ``n_regimes - 1`` change points.
    excl_factor : int, default=5
        Exclusion factor used by FLUSS to avoid selecting nearby change points.

    References
    ----------
    .. [1] Gharghabi, S., Ding, Y., Yeh, C.-C. M., Kamgar, K., Ulanova, L.,
       & Keogh, E. (2017). Matrix Profile VIII: Domain Agnostic Online Semantic
       Segmentation at Superhuman Performance Levels. In *IEEE International
       Conference on Data Mining (ICDM)* (pp. 117-126).

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> from sktime.detection._fluss import FLUSSSegmenter  # doctest: +SKIP
    >>> X = pd.DataFrame({"x": list(range(50))})  # doctest: +SKIP
    >>> est = FLUSSSegmenter(window_length=5, n_regimes=2)  # doctest: +SKIP
    >>> est.fit_predict(X)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["HimanshuBairwa"],
        "maintainers": ["HimanshuBairwa"],
        "python_dependencies": "stumpy",
        # estimator type
        # --------------
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
        "capability:missing_data": False,
        "fit_is_empty": True,
        "X_inner_mtype": "pd.DataFrame",
    }

    def __init__(self, window_length, n_regimes, excl_factor=5):
        self.window_length = window_length
        self.n_regimes = n_regimes
        self.excl_factor = excl_factor
        super().__init__()

    def _predict(self, X):
        """Detect change points in ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Input time series.  Only the first column is used.

        Returns
        -------
        pd.DataFrame
            Sparse format expected by ``BaseDetector`` for
            ``change_point_detection``.  Contains an ``"ilocs"`` column with
            the integer locations of detected change points.
        """
        import numpy as np
        import stumpy

        series = X.iloc[:, 0]
        values = series.to_numpy(dtype=np.float64)

        if len(values) <= self.window_length:
            raise ValueError(
                "window_length must be smaller than the number of "
                "time points in X."
            )

        mp = stumpy.stump(values, m=self.window_length)

        _, _, regime_locations = stumpy.fluss(
            mp,
            L=self.window_length,
            n_regimes=self.n_regimes,
            excl_factor=self.excl_factor,
        )

        valid_locs = []
        if len(regime_locations) > 0:
            for loc in regime_locations:
                if not np.isnan(loc):
                    i = int(loc)
                    if 0 <= i < len(X):
                        valid_locs.append(i)

        return pd.DataFrame({"ilocs": valid_locs})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or
            ``MyClass(**params[i])`` creates a valid test instance.
            ``create_test_instance`` uses the first (or only) dictionary
            in ``params``.
        """
        params0 = {"window_length": 3, "n_regimes": 2, "excl_factor": 1}
        params1 = {"window_length": 4, "n_regimes": 3, "excl_factor": 1}
        return [params0, params1]
