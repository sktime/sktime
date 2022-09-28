# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements simple forecasts based on naive assumptions."""

__all__ = ["NaiveForecaster", "NaiveVariance"]
__author__ = [
    "mloning",
    "piyush1729",
    "sri1419",
    "Flix6x",
    "aiwalter",
    "IlyasMoutawwakil",
    "fkiraly",
    "bethrice44",
]

from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import norm

from sktime.datatypes._convert import convert, convert_to
from sktime.datatypes._utilities import get_slice
from sktime.forecasting.base._base import DEFAULT_ALPHA, BaseForecaster
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.utils.validation import check_window_length
from sktime.utils.validation.forecasting import check_sp


class NaiveForecaster(_BaseWindowForecaster):
    """Forecast based on naive assumptions about past trends continuing.

    NaiveForecaster is a forecaster that makes forecasts using simple
    strategies. Two out of three strategies are robust against NaNs. The
    NaiveForecaster can also be used for multivariate data and it then
    applies internally the ColumnEnsembleForecaster, so each column
    is forecasted with the same strategy.

    Internally, this forecaster does the following:
    - obtains the so-called "last window", a 1D array that denotes the
      most recent time window that the forecaster is allowed to use
    - reshapes the last window into a 2D array according to the given
      seasonal periodicity (prepended with NaN values to make it fit);
    - make a prediction for each column, using the given strategy:
      - "last": last non-NaN row
      - "mean": np.nanmean over rows
    - tile the predictions using the seasonal periodicity

    To compute prediction quantiles, we first estimate the standard error
    of prediction residuals under the assumption of uncorrelated residuals.
    The forecast variance is then computed by multiplying the residual
    variance by a constant. This constant is a small-sample bias adjustment
    and each method (mean, last, drift) have different formulas for computing
    the constant. These formulas can be found in the Forecasting:
    Principles and Practice textbook (Table 5.2) [1]_. Lastly, under the assumption that
    residuals follow a normal distribution, we use the forecast variance and
    z-scores of a normal distribution to estimate the prediction quantiles.

    Parameters
    ----------
    strategy : {"last", "mean", "drift"}, default="last"
        Strategy used to make forecasts:

        * "last":   (robust against NaN values)
                    forecast the last value in the
                    training series when sp is 1.
                    When sp is not 1,
                    last value of each season
                    in the last window will be
                    forecasted for each season.
        * "mean":   (robust against NaN values)
                    forecast the mean of last window
                    of training series when sp is 1.
                    When sp is not 1, mean of all values
                    in a season from last window will be
                    forecasted for each season.
        * "drift":  (not robust against NaN values)
                    forecast by fitting a line between the
                    first and last point of the window and
                    extrapolating it into the future.

    sp : int, or None, default=1
        Seasonal periodicity to use in the seasonal forecasting. None=1.

    window_length : int or None, default=None
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting:
        principles and practice, 3rd edition, OTexts: Melbourne, Australia.
        OTexts.com/fpp3. Accessed on 22 September 2022.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> forecaster.fit(y)
    NaiveForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "scitype:y": "univariate",
        "capability:pred_var": True,
        "capability:pred_int": True,
    }

    def __init__(self, strategy="last", window_length=None, sp=1):
        super(NaiveForecaster, self).__init__()
        self.strategy = strategy
        self.sp = sp
        self.window_length = window_length

        # Override tag for handling missing data
        # todo: remove if GH1367 is fixed
        if self.strategy in ("last", "mean"):
            self.set_tags(**{"handles-missing-data": True})

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        # X_train is ignored
        sp = self.sp or 1

        n_timepoints = y.shape[0]

        if self.strategy in ("last", "mean"):
            # check window length is greater than sp for seasonal mean or seasonal last
            if self.window_length is not None and sp != 1:
                if self.window_length < sp:
                    raise ValueError(
                        f"The `window_length`: "
                        f"{self.window_length} is smaller than "
                        f"`sp`: {sp}."
                    )
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            self.sp_ = check_sp(sp)

            #  if not given, set default window length
            if self.window_length is None:
                self.window_length_ = len(y)

        elif self.strategy == "drift":
            if sp != 1:
                warn("For the `drift` strategy, the `sp` value will be ignored.")
            # window length we need for forecasts is just the
            # length of seasonal periodicity
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            if self.window_length is None:
                self.window_length_ = len(y)
            if self.window_length == 1:
                raise ValueError(
                    f"For the `drift` strategy, "
                    f"the `window_length`: {self.window_length} "
                    f"value must be greater than one."
                )

        else:
            allowed_strategies = ("last", "mean", "drift")
            raise ValueError(
                f"Unknown strategy: {self.strategy}. Expected "
                f"one of: {allowed_strategies}."
            )

        # check window length
        if self.window_length_ > len(self._y):
            param = "sp" if self.strategy == "last" and sp != 1 else "window_length_"
            raise ValueError(
                f"The {param}: {self.window_length_} is larger than "
                f"the training series."
            )

        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Calculate predictions for use in predict."""
        last_window, _ = self._get_last_window()
        fh = fh.to_relative(self.cutoff)

        strategy = self.strategy
        sp = self.sp or 1

        # if last window only contains missing values, return nan
        if np.all(np.isnan(last_window)) or len(last_window) == 0:
            return self._predict_nan(fh)

        elif strategy == "last" or (strategy == "drift" and self.window_length_ == 1):
            if sp == 1:
                last_valid_value = last_window[
                    (~np.isnan(last_window))[0::sp].cumsum().argmax()
                ]
                return np.repeat(last_valid_value, len(fh))

            else:
                # reshape last window, one column per season
                last_window = self._reshape_last_window_for_sp(last_window)

                # select last non-NaN row for every column
                y_pred = last_window[
                    (~np.isnan(last_window)).cumsum(0).argmax(0).T,
                    range(last_window.shape[1]),
                ]

                # tile prediction according to seasonal periodicity
                y_pred = self._tile_seasonal_prediction(y_pred, fh)

        elif strategy == "mean":
            if sp == 1:
                return np.repeat(np.nanmean(last_window), len(fh))

            else:
                # reshape last window, one column per season
                last_window = self._reshape_last_window_for_sp(last_window)

                # compute seasonal mean, averaging over non-NaN rows for every column
                y_pred = np.nanmean(last_window, axis=0)

                # tile prediction according to seasonal periodicity
                y_pred = self._tile_seasonal_prediction(y_pred, fh)

        elif strategy == "drift":
            if self.window_length_ != 1:
                if np.any(np.isnan(last_window[[0, -1]])):
                    raise ValueError(
                        f"For {strategy},"
                        f"first and last elements in the last "
                        f"window must not be a missing value."
                    )
                else:
                    # formula for slope
                    slope = (last_window[-1] - last_window[0]) / (
                        self.window_length_ - 1
                    )

                    # get zero-based index by subtracting the minimum
                    fh_idx = fh.to_indexer(self.cutoff)

                    # linear extrapolation
                    y_pred = last_window[-1] + (fh_idx + 1) * slope
        else:
            raise ValueError(f"unknown strategy {strategy} provided to NaiveForecaster")

        return y_pred

    def _reshape_last_window_for_sp(self, last_window):
        """Reshape the 1D last window into a 2D last window, prepended with NaN values.

        The 2D array has 1 column per season.

        For example:

            last_window = [1, 2, 3, 4]
            sp = 3  # i.e. 3 distinct seasons
            reshaped_last_window = [[nan, nan, 1],
                                    [  2,   3, 4]]
        """
        # if window length is not multiple of sp, backward fill window with nan values
        remainder = self.window_length_ % self.sp_
        if remainder > 0:
            pad_width = self.sp_ - remainder
        else:
            pad_width = 0

        pad_width += self.window_length_ - len(last_window)

        last_window = np.hstack([np.full(pad_width, np.nan), last_window])

        # reshape last window, one column per season
        last_window = last_window.reshape(
            int(np.ceil(self.window_length_ / self.sp_)), self.sp_
        )

        return last_window

    def _tile_seasonal_prediction(self, y_pred, fh):
        """Tile a prediction to cover all requested forecasting horizons.

        The original prediction has 1 value per season.

        For example:

            fh = [1, 2, 3, 4, 5, 6, 7]
            y_pred = [2, 3, 1]  # note len(y_pred) = sp
            y_pred_tiled = [2, 3, 1, 2, 3, 1, 2]
        """
        # we need to replicate the last window if max(fh) is
        # larger than sp,
        # so that we still make forecasts by repeating the
        # last value for that season,
        # assume fh is sorted, i.e. max(fh) == fh[-1]
        # only slicing all the last seasons into last_window
        if fh[-1] > self.sp_:
            reps = int(np.ceil(fh[-1] / self.sp_))
            y_pred = np.tile(y_pred, reps=reps)

        # get zero-based index by subtracting the minimum
        fh_idx = fh.to_indexer(self.cutoff)
        return y_pred[fh_idx]

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        """
        y_pred = super(NaiveForecaster, self)._predict(fh=fh, X=X)

        # test_predict_time_index_in_sample_full[ForecastingPipeline-0-int-int-True]
        #   causes a pd.DataFrame to appear as y_pred, which upsets the next lines
        #   reasons are unclear, this is coming from the _BaseWindowForecaster
        # todo: investigate this
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.iloc[:, 0]

        # check for in-sample prediction, if first time point needs to be imputed
        if self._y.index[0] in y_pred.index:
            if y_pred.loc[[self._y.index[0]]].hasnans:
                # fill NaN with observed values
                y_pred.loc[self._y.index[0]] = self._y[self._y.index[1]]

        return y_pred

    def _predict_quantiles(self, fh, X=None, alpha=0.5):
        """Compute/return prediction quantiles for a forecast.

        Uses normal distribution as predictive distribution to compute the
        quantiles.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        alpha : float or list of float, optional (default=0.5)
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        y_pred = self.predict(fh)
        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")

        pred_var = self.predict_var(fh)
        z_scores = norm.ppf(alpha)

        errors = (
            np.sqrt(pred_var.to_numpy().reshape(len(pred_var), 1)) * z_scores
        ).reshape(len(y_pred), len(alpha))

        pred_quantiles = pd.DataFrame(
            errors + y_pred.values.reshape(len(y_pred), 1),
            columns=pd.MultiIndex.from_product([["Quantiles"], alpha]),
            index=fh.to_absolute(self.cutoff).to_pandas(),
        )

        return pred_quantiles

    def _predict_var(self, fh, X=None, cov=False):
        """Compute/return prediction variance for naive forecasts.

        Variance are computed according to formulas from (Table 5.2)
        Forecasting: Principles and Practice textbook [1]_.

        Must be run *after* the forecaster has been fitted.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        cov : bool, optional (default=False)
            If True, return the covariance matrix.
            If False, return the marginal variance.

        Returns
        -------
        pred_var :
            if cov=False, pd.DataFrame with index fh.
                a vector of same length as fh with predictive marginal variances;
            if cov=True, pd.DataFrame with index fh and columns fh.
                a square matrix of size len(fh) with predictive covariance matrix.

        References
        ----------
        .. [1] https://otexts.com/fpp3/prediction-intervals.html#benchmark-methods
        """
        y = self._y
        y = convert_to(y, "pd.Series")
        T = len(y)

        # Compute "past" residuals
        if self.strategy == "last":
            y_res = y - y.shift(self.sp)
        elif self.strategy == "mean":
            # Since this strategy returns a constant, just predict fh=1 and
            # transform the constant into a repeated array
            y_pred = np.repeat(np.squeeze(self.predict(fh=1)), T)
            y_res = y - y_pred
        else:
            # Slope equation from:
            # https://otexts.com/fpp3/simple-methods.html#drift-method
            slope = (y.iloc[-1] - y.iloc[0]) / (T - 1)

            # Fitted value = previous value + slope
            # https://github.com/robjhyndman/forecast/blob/master/R/naive.R#L34
            y_res = y - (y.shift(self.sp) + slope)

        # Residuals MSE and SE
        # + 1 degrees of freedom to estimate drift coefficient standard error
        # https://github.com/robjhyndman/forecast/blob/master/R/naive.R#L79
        n_nans = np.sum(pd.isna(y_res))
        mse_res = np.sum(np.square(y_res)) / (T - n_nans - (self.strategy == "drift"))
        se_res = np.sqrt(mse_res)

        sp = self.sp
        window_length = self.window_length or T
        # Formulas from:
        # https://otexts.com/fpp3/prediction-intervals.html (Table 5.2)
        partial_se_formulas = {
            "last": lambda h: np.sqrt(h)
            if sp == 1
            else np.sqrt(np.floor((h - 1) / sp) + 1),
            "mean": lambda h: np.repeat(np.sqrt(1 + (1 / window_length)), len(h)),
            "drift": lambda h: np.sqrt(h * (1 + (h / (T - 1)))),
        }

        fh_periods = np.array(fh.to_relative(self.cutoff))
        marginal_se = se_res * partial_se_formulas[self.strategy](fh_periods)
        marginal_vars = marginal_se**2

        fh_idx = fh.to_absolute(self.cutoff).to_pandas()
        if cov:
            fh_size = len(fh)
            cov_matrix = np.fill_diagonal(
                np.zeros(shape=(fh_size, fh_size)), marginal_vars
            )
            pred_var = pd.DataFrame(cov_matrix, columns=fh_idx, index=fh_idx)
        else:
            pred_var = pd.DataFrame(marginal_vars, index=fh_idx)

        return pred_var

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params_list = [
            {},
            {"strategy": "mean", "sp": 2},
            {"strategy": "drift"},
            {"strategy": "last"},
            {"strategy": "mean", "window_length": 5},
        ]

        return params_list


class NaiveVariance(BaseForecaster):
    r"""Compute the prediction variance based on a naive strategy.

    NaiveVariance adds to a `forecaster` the ability to compute the
    prediction variance based on naive assumptions about the time series.
    The simple strategy is as follows:
    - Let :math:`y_1,\dots,y_T` be the time series we fit the estimator :math:`f` to.
    - Let :math:`\widehat{y}_{ij}` be the forecast for time point :math:`j`, obtained
    from fitting the forecaster to the partial time series :math:`y_1,\dots,y_i`.
    - We compute the residuals matrix :math:`R=(r_{ij})=(y_j-\widehat{y}_{ij})`.
    - The variance prediction :math:`v_k` for :math:`y_{T+k}` is
    :math:`\frac{1}{T-k}\sum_{i=1}^{T-k} a_{i,i+k}^2`
    because we are averaging squared residuals of all forecasts that are :math:`k`
    time points ahead.
    - And for the covariance matrix prediction, the formula becomes
    :math:`Cov(y_k, y_l)=\frac{\sum_{i=1}^N \hat{r}_{k,k+i}*\hat{r}_{l,l+i}}{N}`.

    The resulting forecaster will implement
        `predict_interval`, `predict_quantiles`, `predict_var`, and `predict_proba`,
        even if the wrapped forecaster `forecaster` did not have this capability;
        for point forecasts (`predict`), behaves like the wrapped forecaster.

    Parameters
    ----------
    forecaster : estimator
        Estimator to which probabilistic forecasts are being added
    initial_window : int, optional, default=1
        number of minimum initial indices to use for fitting when computing residuals
    verbose : bool, optional, default=False
        whether to print warnings if windows with too few data points occur

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster, NaiveVariance
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> variance_forecaster = NaiveVariance(forecaster)
    >>> variance_forecaster.fit(y)
    NaiveVariance(...)
    >>> var_pred = variance_forecaster.predict_var(fh=[1,2,3])
    """

    _tags = {
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_var": True,
    }

    def __init__(self, forecaster, initial_window=1, verbose=False):

        self.forecaster = forecaster
        self.initial_window = initial_window
        self.verbose = verbose
        super(NaiveVariance, self).__init__()

        tags_to_clone = [
            "requires-fh-in-fit",
            "ignores-exogeneous-X",
            "handles-missing-data",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.forecaster, tags_to_clone)

    def _fit(self, y, X=None, fh=None):

        self.fh_early_ = fh is not None
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y, X=X, fh=fh)

        if self.fh_early_:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y=y, X=X, forecaster=self.forecaster, initial_window=self.initial_window
            )

        return self

    def _predict(self, fh, X=None):
        return self.forecaster_.predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        self.forecaster_.update(y, X, update_params=update_params)
        if update_params and self._fh is not None:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y=self._y,
                X=self._X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
            )
        return self

    def _predict_quantiles(self, fh, X=None, alpha=0.5):
        """Compute/return prediction quantiles for a forecast.

        Uses normal distribution as predictive distribution to compute the
        quantiles.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : float or list of float, optional (default=0.5)
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        y_pred = self.predict(fh, X)
        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")
        pred_var = self.predict_var(fh, X)
        pred_var = pred_var[pred_var.columns[0]]
        pred_var.index = y_pred.index

        z_scores = norm.ppf(alpha)
        errors = [pred_var**0.5 * z for z in z_scores]

        index = pd.MultiIndex.from_product([["Quantiles"], alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        for a, error in zip(alpha, errors):
            pred_quantiles[("Quantiles", a)] = y_pred + error

        fh_absolute = fh.to_absolute(self.cutoff)
        pred_quantiles.index = fh_absolute

        return pred_quantiles

    def _predict_var(self, fh, X=None, cov=False):
        """Compute/return prediction variance for a forecast.

        Must be run *after* the forecaster has been fitted.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            If True, return the covariance matrix.
            If False, return the marginal variance.

        Returns
        -------
        pred_var :
            if cov=False, pd.DataFrame with index fh.
                a vector of same length as fh with predictive marginal variances;
            if cov=True, pd.DataFrame with index fh and columns fh.
                a square matrix of size len(fh) with predictive covariance matrix.
        """
        if self.fh_early_:
            residuals_matrix = self.residuals_matrix_
        else:
            residuals_matrix = self._compute_sliding_residuals(
                y=self._y,
                X=self._X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
            )

        fh_relative = fh.to_relative(self.cutoff)
        fh_absolute = fh.to_absolute(self.cutoff)

        if cov:
            fh_size = len(fh)
            covariance = np.zeros(shape=(len(fh), fh_size))
            for i in range(fh_size):
                i_residuals = np.diagonal(residuals_matrix, offset=fh_relative[i])
                for j in range(i, fh_size):  # since the matrix is symmetric
                    j_residuals = np.diagonal(residuals_matrix, offset=fh_relative[j])
                    max_residuals = min(len(j_residuals), len(i_residuals))
                    covariance[i, j] = covariance[j, i] = np.nanmean(
                        i_residuals[:max_residuals] * j_residuals[:max_residuals]
                    )
            pred_var = pd.DataFrame(
                covariance,
                index=fh_absolute,
                columns=fh_absolute,
            )
        else:
            variance = [
                np.nanmean(np.diagonal(residuals_matrix, offset=offset) ** 2)
                for offset in fh_relative
            ]
            if hasattr(self._y, "columns"):
                columns = self._y.columns
                pred_var = pd.DataFrame(variance, columns=columns, index=fh_absolute)
            else:
                pred_var = pd.DataFrame(variance, index=fh_absolute)

        return pred_var

    def _compute_sliding_residuals(self, y, X, forecaster, initial_window):
        """Compute sliding residuals used in uncertainty estimates.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            sktime compatible time series to use in computing residuals matrix
        X : pd.DataFrame
            sktime compatible exogeneous time series to use in forecasts
        forecaster : sktime compatible forecaster
            forecaster to use in computing the sliding residuals
        initial_window : int
            minimum length of initial window to use in fitting

        Returns
        -------
        residuals_matrix : pd.DataFrame, row and column index = y.index[initial_window:]
            [i,j]-th entry is signed residual of forecasting y.loc[j] from y.loc[:i],
            using a clone of the forecaster passed through the forecaster arg
        """
        y = convert_to(y, "pd.Series")

        y_index = y.index[initial_window:]
        residuals_matrix = pd.DataFrame(columns=y_index, index=y_index, dtype="float")

        for id in y_index:
            forecaster = forecaster.clone()
            y_train = get_slice(y, start=None, end=id)  # subset on which we fit
            y_test = get_slice(y, start=id, end=None)  # subset on which we predict
            try:
                forecaster.fit(y_train, fh=y_test.index)
            except ValueError:
                if self.verbose:
                    warn(
                        f"Couldn't fit the model on "
                        f"time series window length {len(y_train)}.\n"
                    )
                continue
            try:
                residuals_matrix.loc[id] = forecaster.predict_residuals(y_test, X)
            except IndexError:
                warn(
                    f"Couldn't predict after fitting on time series of length \
                     {len(y_train)}.\n"
                )

        return residuals_matrix

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params_list = {"forecaster": FORECASTER}

        return params_list
