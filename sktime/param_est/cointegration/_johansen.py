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

    Attributes
    ----------
    cvm_ :  np.ndarray of float containing critical values
        (90%, 95%, 99%) of maximum eigenvalue statistic

    cvt_ :  np.ndarray of float containing critical values
        (90%, 95%, 99%) of trace statistic

    eig_ :  np.ndarray of float containing eigenvalues of VECM coefficient matrix

    evec_ : np.ndarray of float containing eigenvectors of VECM coefficient matrix

    ind_ : np.ndarray of int containing Order of eigenvalues

    lr1_ : np.ndarray of float containing trace statistic

    lr2_ : np.ndarray of float containing maximum eigenvalue statistic

    max_eig_stat_ : np.ndarray of float containing maximum eigenvalue statistic
        (Needs to be tested, because it seems to be a duplicate in statsmodels)

    max_eig_stat_crit_vals_ : np.ndarray of float containing critical values
        (90%, 95%, 99%) of maximum eigenvalue statistic

    meth_ : str containing the name of the test method

    r0t_ : np.ndarray of float containing residuals for delta Y

    rkt_ : np.ndarray of float containing residuals for delta Y-1

    trace_stat_ : np.ndarray of float containing trace statistics

    trace_stat_crit_vals_ : np.ndarray of float containing critical values
        (90%, 95%, 99%) of trace statistic

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.cointegration import JohansenCointegration
    >>> import pandas as pd
    >>> X = load_airline()
    >>> X2 = X.shift(1).bfill()
    >>> df = pd.DataFrame({"X":X, "X2": X2})
    >>> coint_est = JohansenCointegration()
    >>> coint_est.fit(df)
    >>> print(coint_est.get_fitted_params()["cvm"])
    [[15.0006 17.1481 21.7465]
     [ 2.7055  3.8415  6.6349]]

    Notes
    -----
    The underlying test is a wrapper for the statsmodels cointegration test.
    The max rank (depending on preferred sig-level) needs to be derived from
    the param estimates and be used as coint-rank.  for the other parameters,
    it is advised to choose the same det_order as in the main model.
    Same goes for k_ar_diff and max lag determined. Further, keep in mind,
    X in this case needs to be a minimum of two times series, where X may equal x AND y.
    VECM do not have a classical X and a y series. Both are to be considered endogenous.

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
        X : array_like, e.g. pd.Series
        Contains the full set of time-series to be investigated, all X AND y.
        In VECM typically X and a y do not exist. All X and y are considered endogenous.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        cojo_res = coint_johansen(
            endog=X, det_order=self.det_order, k_ar_diff=self.k_ar_diff
        )

        # Critical values (90%, 95%, 99%) of maximum eigenvalue statistic.
        self.cvm_ = cojo_res.cvm

        # Critical values (90%, 95%, 99%) of trace statistic
        self.cvt_ = cojo_res.cvt

        # Eigenvalues of VECM coefficient matrix
        self.eig_ = cojo_res.eig

        # Eigenvectors of VECM coefficient matrix
        self.evec_ = cojo_res.evec

        # Order of eigenvalues
        self.ind_ = cojo_res.ind

        # Trace statistic
        self.lr1_ = cojo_res.lr1

        # Maximum eigenvalue statistic
        self.lr2_ = cojo_res.lr2

        # Maximum eigenvalue statistic / correct?
        self.max_eig_stat_ = cojo_res.max_eig_stat

        # Critical values (90%, 95%, 99%) of maximum eigenvalue statistic.
        self.max_eig_stat_crit_vals_ = cojo_res.max_eig_stat_crit_vals

        # Test method
        self.meth_ = cojo_res.meth

        # Residuals for delta Y
        self.r0t_ = cojo_res.r0t

        # Residuals for delta Y-1
        self.rkt_ = cojo_res.rkt

        # Trace statistic
        self.trace_stat_ = cojo_res.trace_stat

        # Critical values (90%, 95%, 99%) of trace statistic
        self.trace_stat_crit_vals_ = cojo_res.trace_stat_crit_vals

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
