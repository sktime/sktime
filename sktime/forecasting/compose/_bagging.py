#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements Bagging Forecaster."""

__author__ = ["ltsaprounis"]

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.datatypes._utilities import update_data
from sktime.forecasting.base import BaseForecaster
from sktime.transformations.base import BaseTransformer

PANDAS_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


class BaggingForecaster(BaseForecaster):
    """Forecast a time series by aggregating forecasts from its bootstraps.

    Bagged "Bootstrap Aggregating" Forecasts are obtained by forecasting bootstrapped
    time series and then aggregating the resulting forecasts. For the point forecast,
    the different forecasts are aggregated using the mean function [1]. Prediction
    intervals and quantiles are calculated for each time point in the forecasting
    horizon by calculating the sampled forecast quantiles.

    Bergmeir et al. (2016) [2] show that, on average, bagging ETS forecasts gives better
    forecasts than just applying ETS directly. The default bootstrapping transformer
    and forecaster are selected as in [2].

    Parameters
    ----------
    bootstrap_transformer : sktime transformer BaseTransformer descendant instance
        (default = sktime.transformations.bootstrap.STLBootstrapTransformer)
        Bootstrapping Transformer that takes a series (with tag
        scitype:transform-input=Series) as input and returns a panel (with tag
        scitype:transform-input=Panel) of bootstrapped time series if not specified
        sktime.transformations.bootstrap.STLBootstrapTransformer is used.
    forecaster : sktime forecaster, BaseForecaster descendant instance, optional
        (default = sktime.forecating.ets.AutoETS)
        If not specified, sktime.forecating.ets.AutoETS is used.
    sp: int (default=2)
        Seasonal period for default Forecaster and Transformer. Must be 2 or greater.
        Ignored for the bootstrap_transformer and forecaster if they are specified.
    random_state: int or np.random.RandomState (default=None)
        The random state of the estimator, used to control the random number generator

    See Also
    --------
    sktime.transformations.bootstrap.MovingBlockBootstrapTransformer :
        Transformer that applies the Moving Block Bootstrapping method to create
        a panel of synthetic time series.

    sktime.transformations.bootstrap.STLBootstrapTransformer :
        Transformer that utilises BoxCox, STL and Moving Block Bootstrapping to create
        a panel of similar time series.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3,
        Chapter 12.5. Accessed on February 13th 2022.
    .. [2] Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential
        smoothing methods using STL decomposition and Box-Cox transformation.
        International Journal of Forecasting, 32(2), 303-312

    Examples
    --------
    >>> from sktime.transformations.bootstrap import STLBootstrapTransformer
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import BaggingForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = BaggingForecaster(
    ...     STLBootstrapTransformer(sp=12), NaiveForecaster(sp=12)
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    BaggingForecaster(...)
    >>> y_hat = forecaster.predict([1,2,3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["ltsaprounis", "fkiraly"],
        "scitype:y": "both",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "y_inner_mtype": PANDAS_MTYPES,
        # which types do _fit, _predict, assume for y?
        "X_inner_mtype": PANDAS_MTYPES,
        # which types do _fit, _predict, assume for X?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "requires-fh-in-fit": False,  # like AutoETS overwritten if forecaster not None
        "enforce_index_type": None,  # like AutoETS overwritten if forecaster not None
        "capability:insample": True,  # can the estimator make in-sample predictions?
        "capability:pred_int": True,  # can the estimator produce prediction intervals?
        "capability:pred_int:insample": True,  # ... for in-sample horizons?
    }

    def __init__(
        self,
        bootstrap_transformer: BaseTransformer = None,
        forecaster: BaseForecaster = None,
        sp: int = 2,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.bootstrap_transformer = bootstrap_transformer
        self.forecaster = forecaster
        self.sp = sp
        self.random_state = random_state

        if bootstrap_transformer is None:
            # if the transformer is None, this uses the statsmodels dependent
            # sktime.transformations.bootstrap.STLBootstrapTransformer
            #
            # done before the super call to trigger exceptions
            self.set_tags(**{"python_dependencies": "statsmodels"})

        super().__init__()

        # set the tags based on forecaster
        tags_to_clone = [
            "requires-fh-in-fit",  # is forecasting horizon already required in fit?
            "enforce_index_type",
        ]
        if forecaster is not None:
            self.clone_tags(self.forecaster, tags_to_clone)

        self.bootstrap_transformer_ = self._check_transformer(bootstrap_transformer)
        self.forecaster_ = self._check_forecaster(forecaster)

    def _check_transformer(self, transformer):
        """Check if the transformer is a valid transformer for BaggingForecaster.

        Also replaces with default if transformer is None

        Parameters
        ----------
        transformer : BaseTransformer
            The transformer to check

        Returns
        -------
        fresh clone of the transformer to set to self.bootstrap_transformer_
        """
        from sktime.registry import scitype

        if transformer is None:
            from sktime.transformations.bootstrap import STLBootstrapTransformer

            return STLBootstrapTransformer(sp=self.sp)

        msg = (
            "Error in BaggingForecaster: "
            "bootstrap_transformer in BaggingForecaster should be an sktime transformer"
            " that takes as input a Series and output a Panel."
        )

        t_inp = transformer.get_tag("scitype:transform-input", raise_error=False)
        t_out = transformer.get_tag("scitype:transform-output", raise_error=False)

        if t_inp != "Series" or t_out != "Panel":
            raise TypeError(msg)
        if scitype(transformer) != "transformer":
            raise TypeError(msg)

        if hasattr(transformer, "clone"):
            return transformer.clone()
        else:
            from sklearn import clone

            return clone(transformer)

    def _check_forecaster(self, forecaster):
        """Check if the forecaster is a valid transformer for BaggingForecaster.

        Also replaces with default if forecaster is None

        Parameters
        ----------
        forecaster : BaseForecaster
            The forecaster to check

        Returns
        -------
        fresh clone of the forecaster to set to self.forecaster_
        """
        from sktime.registry import scitype

        if not scitype(forecaster) == "forecaster":
            raise TypeError(
                "Error in BaggingForecaster: "
                "forecaster in BaggingForecaster should be an sktime forecaster"
            )

        if forecaster is None:
            from sktime.forecasting.ets import AutoETS

            return AutoETS(sp=self.sp)
        else:
            return forecaster.clone()

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # random state handling passed into input estimators
        self.random_state_ = check_random_state(self.random_state)

        # fit/transform the transformer to obtain bootstrap samples
        y_bootstraps = self.bootstrap_transformer_.fit_transform(X=y)
        self._y_bs_ix = y_bootstraps.index

        # generate replicates of exogenous data for bootstrap
        X_inner = self._gen_X_bootstraps(X)

        # fit the forecaster to the bootstrapped samples
        self.forecaster_.fit(y=y_bootstraps, fh=fh, X=X_inner)

        return self

    def _gen_X_bootstraps(self, X, y_bootstraps):
        """Generate replicates of exogenous data for bootstrap.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous time series, non-hierarchical
        y_bootstraps : pd.DataFrame in pd-multiindex format
            Bootstrapped endogenous data

        Returns
        -------
        X_bootstraps : pd.DataFrame
            Bootstrapped exogenous data
        """
        if X is None:
            return None

        if isinstance(X.index, pd.MultiIndex):
            X_nlv = X.index.nlevels
        else:
            X_nlv = 1

        levels_to_drop = list(range(-X_nlv, 0))

        y_bs_ix = self._y_bs_ix

        inst_ix = y_bs_ix.droplevel(levels_to_drop).unique()
        X_repl = [X] * len(inst_ix)
        X_bootstraps = pd.concat(X_repl, keys=inst_ix)
        return X_bootstraps

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        # generate replicates of exogenous data for bootstrap
        X_inner = self._gen_X_bootstraps(X)

        # compute bootstrapped forecasts
        y_bootstraps_pred = self.forecaster_.predict(fh=fh, X=X_inner)

        # aggregate bootstrapped forecasts
        y_pred = y_bootstraps_pred.groupby(level=-1).mean().iloc[:, 0]
        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        # X is ignored
        y_pred = self.forecaster_.predict(fh=fh, X=None)
        return self._calculate_data_quantiles(y_pred, alpha)

    def _update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.array
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        # Need to construct a completely new y out of ol self._y and y and then
        # fit_treansform the transformer and re-fit the forecaster.
        _y = update_data(self._y, y)

        self.bootstrap_transformer_.fit(X=_y)
        y_bootstraps = self.bootstrap_transformer_.transform(X=_y)
        self.forecaster_.fit(y=y_bootstraps, fh=self.fh, X=None)

        return self

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.bootstrap import MovingBlockBootstrapTransformer
        from sktime.utils.estimators import MockForecaster
        from sktime.utils.validation._dependencies import _check_soft_dependencies

        params = [
            {
                "bootstrap_transformer": MovingBlockBootstrapTransformer(),
                "forecaster": MockForecaster(),
            },
        ]

        # the default param set causes a statsmodels based estimator
        # to be created as bootstrap_transformer
        if _check_soft_dependencies("statsmodels", severity="none"):
            params += [{}]

        return params

    def _calculate_data_quantiles(self, df: pd.DataFrame, alpha: List[float]):
        """Generate quantiles for each time point.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe of mtype pd-multiindex or hierarchical
        alpha : List[float]
            list of the desired quantiles

        Returns
        -------
        pd.DataFrame
            The specified quantiles
        """
        var_names = self._get_varnames()
        var_name = var_names[0]

        index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        for a in alpha:
            quant_a = df.groupby(level=-1, as_index=True).quantile(a)
            pred_quantiles[[(var_name, a)]] = quant_a

        return pred_quantiles
