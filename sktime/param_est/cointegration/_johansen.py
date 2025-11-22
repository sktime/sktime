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

    def _fit(self, endog):
        """Fit estimator and estimate parameters from cointegration method.

        As long as a trace statistic is bigger as a critical value
        (dep. on 90, 95 or 99 sig-level), there exists a cointegration up to this level.

        Reading Example for 0-2 cointegration ranks.:
        self.lr1 = [400, 300, 50]
        self.cvt = [[300, 100, 80], [200, 50, 70], [100, 40, 60]]
        Then we have cointegration rank 2 up to the 95 % sig level,
        because 60 > 50, where trace statistic is no longer bigger
        than crit val.

        Parameters
        ----------
        endog : array_like, e.g. pd.Series
        Contains the full set of time-series to be investigated, all X and y.
        In VECM typically X and y do not exist. All X and y are considered endogenous.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        cojo_res = coint_johansen(
            endog=endog, det_order=self.det_order, k_ar_diff=self.k_ar_diff
        )

        # Critical values (90%, 95%, 99%) of maximum eigenvalue statistic.
        self.cvm = cojo_res.cvm

        # Critical values (90%, 95%, 99%) of trace statistic
        self.cvt = cojo_res.cvt

        # Eigenvalues of VECM coefficient matrix
        self.eig = cojo_res.eig

        # Eigenvectors of VECM coefficient matrix
        self.evec = cojo_res.evec

        # Order of eigenvalues
        self.ind = cojo_res.ind

        # Trace statistic
        self.lr1 = cojo_res.lr1

        # Maximum eigenvalue statistic
        self.lr2 = cojo_res.lr2

        # Maximum eigenvalue statistic / correct?
        self.max_eig_stat = cojo_res.max_eig_stat

        # Critical values (90%, 95%, 99%) of maximum eigenvalue statistic.
        self.max_eig_stat_crit_vals = cojo_res.max_eig_stat_crit_vals

        # Test method
        self.meth = cojo_res.meth

        # Residuals for delta Y
        self.r0t = cojo_res.r0t

        # Residuals for delta Y-1
        self.rkt = cojo_res.rkt

        # Trace statistic
        self.trace_stat = cojo_res.trace_stat

        # Critical values (90%, 95%, 99%) of trace statistic
        self.trace_stat_crit_vals = cojo_res.trace_stat_crit_vals

        # need to check for redundancies in results

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
