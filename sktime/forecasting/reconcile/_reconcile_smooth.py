# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Smooth forecast reconciliation for hierarchical data, by Sakai Ando."""

__all__ = ["ReconcilerSmoothForecaster"]
__author__ = ["SakaiAndo"]

import numpy as np
import pandas as pd
from numpy.linalg import inv

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import ExpandingWindowSplitter
from sktime.utils.warnings import warn


class ReconcilerSmoothForecaster(BaseForecaster):
    """Smooth forecast reconciliation for hierarchical data, by Sakai Ando.

    This algorithm reconciles hierarchical forecasts and ensures
    smoothness of reconciled smooth forecasts by integrating
    minimum-trace reconciliation and Hodrick-Prescott filter.

    With linear constraints, the method has a closed-form solution,
    convenient for a high-dimensional environment.

    Refer to the original publication [1]_ for further details.

    Parameters
    ----------
    forecaster : estimator
        Estimator to generate base forecasts which are then reconciled
    n_splits : int, optional (default=10)
        Number of splits for expanding window cross-validation
        in computation of weight matrix
    smoothing : bool, optional (default=True)
        Whether to apply smoothing in the reconciliation step

    See Also
    --------
    Aggregator
    Reconciler
    ReconcilerForecaster

    References
    ----------
    .. [1] Ando S (2024). Smooth Forecast Reconciliation. IMF working paper.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "SakaiAndo",
        "maintainers": "SakaiAndo",
        "python_dependencies": ["sympy"],
        # estimator type
        # --------------
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
        "fit_is_empty": False,
    }

    def __init__(self, forecaster, n_splits=10, smoothing=True):
        self.forecaster = forecaster
        self.n_splits = n_splits
        self.smoothing = smoothing

        super().__init__()

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, default=None
            Exogenous variables for the base forecaster

        Returns
        -------
        self : reference to self
        """
        # handle non-hierarchical data, early return
        if y.index.nlevels < 2:
            self.forecaster_ = self.forecaster.clone()
            self.forecaster_.fit(y=y, X=X, fh=fh)
            return self

        # fit forecasters for each level
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y, X=X, fh=fh)

        fh_relative = fh.to_relative(self.cutoff)

        n_splits = self.n_splits
        forecaster = self.forecaster.clone()
        W = _generate_weight(y, forecaster, fh_relative, n_splits)
        self.W_ = W

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        base_fc = self.forecaster_.predict(fh=fh, X=X)

        # handle non-hierarchical data, early return
        if base_fc.index.nlevels < 2:
            warn(
                "Reconciler is intended for use with y.index.nlevels > 1. "
                "Returning predictions unchanged.",
                obj=self,
            )
            return base_fc

        C, d = _generate_constraints_from_equations(
            [],
            base_fc,
            # TODO - what is this in general?
            ['y_2023Q2 - ' + str(y['2023Q2']),
            'y_2023Q3 + y_2023Q4 -' + str(4*y_A['2023']-y['2023Q1']-y['2023Q2']),
            'y_2024Q1 + y_2024Q2 + y_2024Q3 + y_2024Q4 - ' + str(4*y_A['2024']),
            'y_2025Q1 + y_2025Q2 + y_2025Q3 + y_2025Q4 - ' + str(4*y_A['2025'])
        ])
        W = self.W_

        if self.smoothing:
            recon_fc = _reconcile_smooth(base_fc, W, C, d)
        else:
            recon_fc = _reconcile_only(base_fc, W, C, d)

        return recon_fc

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.compose import YfromX
        from sktime.forecasting.trend import TrendForecaster

        params1 = {"forecaster": TrendForecaster()}  # without exogeneous
        params2 = {"forecaster": YfromX.create_test_instance()}  # with exogeneous
        params3 = {"forecaster": TrendForecaster(), "smoothing": False}  # no smoothing
        params4 = {"forecaster": TrendForecaster(), "n_splits": 5}  # different n_splits

        return [params1, params2, params3, params4]


def _generate_weight(y_hist, forecaster, fh_relative, n_splits):
    """Generate weight matrix for forecast reconciliation from historical data."""
    init_window = y_hist.shape[0] - (n_splits - 1)* 4 - fh_relative.shape[0] -1
    cv = ExpandingWindowSplitter(
        fh=fh_relative,
        step_length=4,
        initial_window=init_window,
    )

    assert n_splits == cv.get_n_splits(y_hist)

    df = evaluate(forecaster=forecaster, y=y_hist, cv=cv, return_data=True)
    fe = np.concatenate(df['y_test'] - df['y_pred']).reshape(n_splits,-1)
    assert fe.shape == (n_splits,fh_relative.shape[0])

    W = _OASD(fe)
    return W 

def _OASD(X):
    """Optimal Average Shrinkage Covariance Estimator."""
    # X: (obs,var) np matrix
    n = X.shape[0]
    S = np.cov(np.transpose(X))
    diagS = np.diag(np.diag(S))
    phi = (np.trace(S@S) - np.trace(diagS@diagS)) / \
            (np.trace(S@S) + np.trace(S) ** 2 - 2 * np.trace(diagS@diagS))
    rho = np.min([1/(n*phi),1])
    S_OASD = (1-rho) * S + rho * diagS
    return S_OASD


def _generate_constraints_from_equations(dfA_fh,dfQ_fh,constraints):
    import sympy as sy

    # make sure df are dataframe
    dfA = pd.DataFrame(dfA_fh)
    dfQ = pd.DataFrame(dfQ_fh)

    # variables, and check time index doesn't include var name
    vA_list = dfA.columns.to_list()
    vQ_list = dfQ.columns.to_list()
    v_list  = vA_list + vQ_list
    for t in [str(t) for t in dfA.index] + [str(t) for t in dfQ.index]:
        for v in v_list:
            if t.find(v) >=0:
                raise ValueError('Do not include '+v+' in time index')
                
    # variables with time
    vAfh = dfA.unstack().index
    vAfh = vAfh.get_level_values(0).astype(str) + '_' + vAfh.get_level_values(1).astype(str)  # noqa: E501
    vAfh = vAfh.to_list()
    vQfh = dfQ.unstack().index
    vQfh = vQfh.get_level_values(0).astype(str) + '_' + vQfh.get_level_values(1).astype(str)  # noqa: E501
    vQfh = vQfh.to_list()
    vfh  = vAfh + vQfh
    
    # clone constraints to all forecast horizons
    long_constraints = []
    for e in constraints:
        if any([e.find(v)>=0 for v in vfh]): # if time is specificied
            long_constraints.append(e) # keep it as it is
        elif any([e.find(v) >= 0 for v in vA_list]): # otherwise, for annual var
            for t in dfA.index:
                e_long = e
                for v in vA_list:
                    e_long = e_long.replace(v,v + '_' + str(t)) # add time after
                long_constraints.append(e_long)
        elif any([e.find(v) >= 0 for v in vQ_list]): # for quarterly var
            for t in dfQ.index:
                e_long = e # create a copy
                for v in vQ_list:
                    e_long = e_long.replace(v,v + '_' + str(t)) # add time after
                long_constraints.append(e_long)
    print(pd.DataFrame(long_constraints))
    print(vfh)
    sy_e = sy.Array(sy.sympify(long_constraints)) # sympy expressions
    sy_v = sy.var(vfh) # sympy variables
    
    # generate numpy matrix and vector
    C = np.array(sy.derive_by_array(sy_e,sy_v),dtype='float').T
    d = -np.array(sy_e.subs(dict(zip(sy_v,np.zeros(len(sy_v))))),dtype='float').reshape(-1,1)  # noqa: E501
    
    return C, d


def _reconcile_only(yhat, W, C, d):
    Cp = np.transpose(C)
    d  = np.array(d).reshape(-1,1)
    index = yhat.index
    yhat = np.array(yhat).reshape(-1,1)
    y_reco = yhat + W @ Cp @ inv(C @ W @ Cp) @ (d - C @ yhat)
    y_reco = pd.Series(y_reco.flatten(), index=index)
    return y_reco


def _HP_matrix(N):
    """Generate HP matrix for 2nd step forecast."""
    F = np.zeros([N,N])
    for i in range(N):
        F[i,i] = 6
    for i in range(N-1):
        F[i+1,i] = -4
        F[i,i+1] = -4
    for i in range(N-2):
        F[i+2,i] = 1
        F[i,i+2] = 1
    F[0,0]   = 1
    F[1,0]   = -2
    F[0,1]   = -2
    F[1,1]   = 5
    F[-1,-1] = 1
    F[-2,-1] = -2
    F[-1,-2] = -2
    F[-2,-2] = 5
    return F

def _reconcile_smooth(yhat, W, C, d):
    N = yhat.shape[0]
    Cp = np.transpose(C)
    d  = np.array(d).reshape(-1,1)
    index = yhat.index
    yhat = np.array(yhat).reshape(-1,1)
    assert C.shape[1] == yhat.shape[0]
    F = _HP_matrix(N)
    lam = 1600
    Phi = lam * F/ ( np.diag(W).min() )
    yhat = np.array(yhat).reshape(-1,1)
    d  = np.array(d).reshape(-1,1)
    y_2nd = ( np.eye(N) - inv( inv(W) + Phi ) @ Cp @ inv( C @ inv( inv(W) + Phi ) @ Cp) @ C ) \  # noqa: E501
        @ inv( inv(W) + Phi ) @ inv(W) @ yhat \
        + inv( inv(W) + Phi ) @ Cp @ inv(C @ inv( inv(W) + Phi ) @ Cp) @ d
    y_2nd = pd.Series(y_2nd.flatten(), index = index)
    return y_2nd
