"""Parameter estimation for univariate/multivariate impulse response function."""

author = ["OldPatrick"]
all = ["ImpulseResponseFunction"]

import warnings

import numpy as np
import pandas as pd

from sktime.param_est.base import BaseParamFitter


class ImpulseResponseFunction(BaseParamFitter):
    """Calculation of Impulse Response Parameters for various time-series forecasters.

    Direct interface to
    ``statsmodels.tsa.statespace.[any_non_var_vecm_model].[MODEL_FROM_MODEL_MAPPING].impulse_responses``
    and
    ``statsmodels.tsa.vector_ar.irf.IRAnalysis``.

    Basically, an impulse reflects a simple input signal into a system. While system
    itself sounds very vague,in the context of time-series such a system can be simply
    a time series itself or a relationship between two time series. Especially in the
    context of time series, such a relationship is often assumed to be linear
    and dynamic and therefore to be found in in linear dynamic models such as VAR and
    VECM, but also in state-space models like Dynamic Factor (ignoring the fact we
    could write all time-series in statespace forms).

    Going further, an impulse response traces how a one-time shock or sudden change
    of one time series variable within a system (of several time-series variables)
    unfolds over time in the whole system of all variables. Practical examples could
    be the shock/change in oil prices on gasoline prices, see for example: Chudik
    and Georgiadis 2019, "Estimation of impulse response functions when shocks are
    observed at a higher frequency than outcome variables.", Working Paper No. 2307,
    European Central Bank.

    The following ``sktime`` estimators support the calculation of an impulse response:

    - ``DynamicFactor``
    - ``VAR``
    - ``VARMAX``
    - ``VECM``

    Parameters
    ----------
    model : Any
        A previous fitted ``sktime`` time series forecaster from ``sktime.forecasting``.
        See above for the current supported ``sktime`` models.

    steps : int, optional, default=1
        The number of steps for which impulse responses are calculated.
        Default is 1. Note that for time-invariant models, the initial
        impulse is not counted as a step, so if steps=1, the output
        will have 2 entries.

    impulse : int, str or array_like, optional, default=0
        If an integer, the state innovation to pulse; must be between 0 and k_posdef-1.
        If a str, it indicates which column of df the unit (1) impulse is given.
        Alternatively, a custom impulse vector may be provided; must be shaped
        k_posdef x 1.

    orthogonalized : bool, optional, default=False
        Whether or not to perform impulse using orthogonalized innovations.
        Note that this will also affect custom impulse vectors.

    cumulative : bool, optional, default=False
        Whether or not to return cumulative impulse responses.

    anchor : int, str, or datetime, optional, default = #start#
        Time point within the sample for the state innovation impulse.
        Type depends on the index of the given endog in the model.
        Two special cases are the strings #start# and #end#, which refer to
        setting the impulse at the first and last points of the sample, respectively.
        Integer values can run from 0 to nobs - 1, or can be negative to apply negative
        indexing. Finally, if a date/time index was provided to the model, then this
        argument can be a date string to parse or a datetime type.

    transformed : bool, optional, default=True
        Whether or not params is already transformed.

    includes_fixed : bool, optional, default=False
        If parameters were previously fixed with the fix_params method, this argument
        describes whether or not params also includes the fixed parameters, in addition
        to the free parameters.

    Attributes
    ----------
    irf_ :  np.ndarray
        Responses for each endogenous variable due to the impulse given by the impulse
        argument. For a time-invariant model, the impulse responses are given for
        steps + 1 elements (this gives the “initial impulse” followed by steps responses
        for the important cases of VAR and SARIMAX models), while for time-varying
        models the impulse responses are only given for steps elements (to avoid having
        to unexpectedly provide updated time-varying matrices). The output from the
        example may be read as follows: (i) Rows show the response variable or the
        variable affected. (ii) Columns show the impulse variable, the variable
        receiving the shock. The t=0, reflects the first dimension, so the immediate
        impact due to change of the impact variable. Impulse Response of statsmodels
        only shows the response for t=0 in this manner. So in the example variable 1
        (X, column 0) receives in t=0 a shock it jumps by 1414 (row 0) and variable 2
        (row 1) by -1.45.If variable 2 (X2, column 1) receives a shock in t=0
        variable 1 jumps by 1401 and variable 2 by -1.45. When orthogonalized=True we
        speak of one std shock, when False of a one unit shock. This explanation holds
        true for alllinear multivariate time-series. If cumulative=True cumulates
        effects up to period t for all responses. If cumulative = False and forecast
        horizon is t=0, then the cumulative = False and True equals.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.impulse import ImpulseResponseFunction
    >>> from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn
    >>> import pandas as pd
    >>> X = load_airline()
    >>> X2 = X.shift(1).bfill()
    >>> df = pd.DataFrame({"X":X, "X2": X2})
    >>> fitted_model = skdyn(k_factors=1, factor_order=2).fit(df)
    >>> sktime_irf = ImpulseResponseFunction(fitted_model, orthogonalized=True)
    >>> sktime_irf.fit(df)
    ImpulseResponseFunction(...)
    >>> print(sktime_irf.get_fitted_params()["irf"])  # doctest: +SKIP
    [[1414.75907225 1401.6016836 ]
     [  -1.45858745   -1.44502246]]

    Notes
    -----
    Parameter and Attribute description taken from statsmodels.Statsmodels has up to
    today two different interfaces for impulse responses. The first one is older and
    seems to serve only VAR, VECM and SVAR models. Within the IRAnalysis class is a
    plotting option showing directly the fade-out of the impulse response signal.
    Since an Impulse Response Function measures the change in a dynamic linear
    relationship, the concept of cointegration plays again a significant role again.

    References
    ----------
    .. [1] Ballarin, G. 2025: Impulse Response Analysis of Structural Nonlinear
    Time Series Models, https://arxiv.org/html/2305.19089v5

    .. [2] Statsmodels (last visited 15/02/2026):
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.impulse_responses.html

    .. [3] Statsmodels (last visited 15/02/2026):
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor.DynamicFactor.impulse_responses.html

    .. [4] Statsmodels (last visited 01/03/2026):
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.irf.IRAnalysis.html
    """

    _tags = {
        "X_inner_mtype": "np.ndarray",  # no support of pl.DataFrame
        "capability:missing_values": False,
        "capability:multivariate": True,
        "capability:pairwise": False,
        "authors": "OldPatrick",
        "python_dependencies": "statsmodels",
        # CI and test flags
        # -----------------
        "tests:skip_by_name": "test_fit_does_not_overwrite_hyper_params",
        # reason for the failure is that deepcopy(self.model) has a different
        # joblib hash from self.model, which erroneously leads the test
        # to believe that the model is not preserved on fit, even though it is.
        # This behaviour is unclear and needs to be investigated further,
        # e.g., why deepcopy(self.model) can even have a different joblib hash.
    }

    def __init__(
        self,
        model=None,  # default fitted None
        steps=1,
        impulse=0,
        orthogonalized=False,
        cumulative=False,
        anchor=None,
        transformed=True,
        includes_fixed=False,
    ):
        self.model = model  # needs a previously fitted model
        self.steps = steps
        self.impulse = impulse
        self.orthogonalized = orthogonalized
        self.cumulative = cumulative
        self.anchor = anchor
        self.transformed = transformed
        self.includes_fixed = includes_fixed

        super().__init__()

    def _fit(self, X, y=None):
        """Fit estimator for univariate and multivariate orthogonal or cumulative irfs.

        Text from statsmodels:
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.impulse_responses.html

        Responses for each endogenous variable due to the impulse given by the
        impulse argument. For a time-invariant model, the impulse responses are
        given for steps + 1 elements (this gives the “initial impulse” followed
        by steps responses for the important cases of VAR and SARIMAX models),
        while for time-varying models the impulse responses are only given for
        steps elements (to avoid having to unexpectedly provide updated
        time-varying matrices). Keep in mind not every model is able to calculate IRF
        for univariate data.

        Parameters
        ----------
        X : array_like, e.g. pd.Series or pd.DataFrame
        Contains the full set of time-series to be investigated, all X AND y.

        y : array_like, e.g. pd.Series or pd.DataFrame, optional (default=None)
        Can be used for additional time-series input influencing the system, but
        not be influenced by the system, e.g. exog. variables like temperature
        or policies. Hint: VECM/VAR exog will be given to the fitted models in
        statsmodels, no need to give it here.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.api import VAR, VECM
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        from statsmodels.tsa.statespace.varmax import VARMAX

        MODEL_MAPPING = {
            "DynamicFactor": DynamicFactor,
            "VARMAX": VARMAX,
            # IRFANALYSIS Methods from here
            "VAR": VAR,
            "VECM": VECM,
        }

        model_name = self.model.__class__.__name__
        sm_wrapper = self.model._fitted_forecaster

        if len(X.shape) < 2 or X.shape[1] < 2:
            # some models have problem with univariate d,ata need warning
            # to show that results can not be calculated univariate,
            # should not be a Problem for ARIMA for instance.
            warnings.warn(
                f"Check if your estimator supports univariate IRF:Got shape {X.shape}."
            )

        if model_name == "VECM" or model_name == "VAR":
            # VECM/VAR has IRF Analsis object. This object has normally
            # more results available, inluding plots, here we limit the
            # reults in order to have a unified return.
            irf_result = sm_wrapper.irf(periods=self.steps)

            if self.orthogonalized and self.cumulative:
                irf_slice = irf_result.orth_cum_effects
            elif self.orthogonalized:
                irf_slice = irf_result.orth_irfs
            elif self.cumulative:
                irf_slice = irf_result.cum_effects
            else:
                irf_slice = irf_result.irfs

            self.irf_ = irf_slice[:, :, self.impulse]
            return self

        ImportedModel = MODEL_MAPPING[model_name]
        k_vars = sm_wrapper.model.k_endog
        fitted_params = sm_wrapper.params
        dummy_data = np.zeros((10, k_vars))
        dummy_model = None

        if model_name == "VARMAX":
            p = sm_wrapper.model.k_ar
            q = sm_wrapper.model.k_ma
            trend_type = sm_wrapper.model.trend
            dummy_model = ImportedModel(dummy_data, order=(p, q), trend=trend_type)

        elif model_name == "DynamicFactor":
            k_factors = sm_wrapper.model.k_factors
            factor_order = sm_wrapper.model.factor_order
            error_order = sm_wrapper.model.error_order

            dummy_model = ImportedModel(
                endog=dummy_data,
                k_factors=k_factors,
                factor_order=factor_order,
                error_order=error_order,
                enforce_stationarity=False,
            )

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        irf_result = dummy_model.impulse_responses(
            params=fitted_params,
            steps=self.steps,
            orthogonalized=self.orthogonalized,
            cumulative=self.cumulative,
            impulse=self.impulse,
            exog=y,
        )

        self.irf_ = irf_result

        return self

    @classmethod
    def _get_clone_plugins(cls):
        """Get clone plugins for BaseObject.

        See scikit-base documentation for details on clone plugins.

        We need to override the cloning functionality for this estimator,
        since the ``model`` attribute is already fitted when passed.
        We do not want to reset ``model`` on ``clone``, and this needs to be
        overridden.

        Returns
        -------
        list of BaseCloner descendants, or None
            List of clone plugins for descendants.
            Each plugin must inherit from ``BaseCloner``
            in ``skbase.base._clone_plugins``, and implement
            the methods ``_check`` and ``_clone``.
        """
        from skbase.base._clone_plugins import _CloneSkbase

        class ModelCloner(_CloneSkbase):
            """Clone plugin to preserve model attribute of self."""

            def _check(self, obj):
                """Check if the plugin should be applied to the given object."""
                return isinstance(obj, ImpulseResponseFunction)

            def _clone(self, obj):
                """Clone the ``model`` attribute of the given object."""
                # we do not want to reset the model on clone, so we return it as is
                temp = obj.model
                clone = super()._clone(obj)
                clone.model = temp
                return clone

        return [ModelCloner]

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
        from sktime.utils.dependencies import _check_soft_dependencies

        if _check_soft_dependencies("statsmodels", severity="none"):
            from sktime.datasets import load_airline
            from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn

            X = load_airline()
            X2 = X.shift(1).bfill()
            df = pd.DataFrame({"X": X, "X2": X2})
            fitted_model = skdyn(k_factors=1, factor_order=2).fit(df)
        else:
            fitted_model = None

        params1 = {
            "model": fitted_model,
            "steps": 1,
            "impulse": 0,
            "orthogonalized": True,
            "cumulative": True,
            "anchor": None,
            "transformed": True,
            "includes_fixed": False,
        }
        params2 = {
            "model": fitted_model,
            "steps": 1,
            "impulse": 0,
            "orthogonalized": False,
            "cumulative": False,
            "anchor": None,
            "transformed": True,
            "includes_fixed": False,
        }

        return [params1, params2]
