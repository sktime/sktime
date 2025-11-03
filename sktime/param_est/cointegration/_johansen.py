# needs more parameters from the original
"""Parameter estimation for cointegration."""

author = ["PBormann"]
all = ["JohansenCointegration"]

from sktime.param_est.base import BaseParamFitter


class JohansenCointegration(BaseParamFitter):
    """Test for cointegration ranks/relationships for VECM Time-Series.

    Direct interface to ``statsmodels.tsa.vector_ar.vecm``.

    Determines the coint parameter value to be used in VECM time-series module vecm.py:
    `coint_rank`. Uses trace statistics or eigenvalue.


    Parameters
    ----------
    endog : array_like, e.g. pd.Series
        Contains the full set of time-series to be investigated, all X and y.
        In VECM typically X and y do not exist. All X and y are considered endogenous.
    det_order : int, default=1
        * -1 - no deterministic terms
        *  0 - constant term
        *  1 - linear trend
    k_ar_diff : int, nonnegative, default=1
        Number of lagged differences in the model. Needs multivariate version of
        ARLagOrderSelector, See also: statsmodels.tsa.vector_ar.vecm.select_order

    Returns
    -------
    result : JohansenTestResult
        An object containing the test's results. The most important attributes
        of the result class are:

        * trace_stat and trace_stat_crit_vals
        * max_eig_stat and max_eig_stat_crit_vals

    Notes
    -----
    The underlying test is a wrapper for the statsmodels cointegration test.
    The max rank (depending on preferred sig-level) needs to be derived from
    the param estimates and be used as coint<-rank.  for the other parameters,
    it is advised to choose the same det_order as in the main model.
    Same goes for k_ar_diff and max lag determined.

    Reading Example
    ---------------
    tbd

    See Also
    --------
    statsmodels.tsa.vector_ar.vecm.select_coint_rank

    References
    ----------
    .. [1] Lütkepohl, H. 2005. New Introduction to Multiple Time Series
        Analysis. Springer.
    .. [2] Statsmodels (last visited 02/11/2025):
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
    """

    _tags = {
        "X_inner_mtype": "np.ndarray",  # no support of pl.DataFrame
        "capability:missing_values": False,
        "capability:multivariate": True,
        "capability:pairwise": True,
        "authors": "PBormann",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        det_order=1,
        k_ar_diff=1,
    ):
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff

        super().__init__()

    def _fit(self, X):
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        cojo_res = coint_johansen(
            endog=X, det_order=self.det_order, k_ar_diff=self.k_ar_diff
        )

        self.trace_stat = cojo_res.trace_stat
        self.trace_stat_crit_vals = cojo_res.trace_stat_crit_vals

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator/test.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "det_order": 1,
            "k_ar_diff": 1,
        }

        return [params1, params2]
