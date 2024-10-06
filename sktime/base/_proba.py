# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements mixin class for probabilistic prediction defaulting behaviour."""

__author__ = ["fkiraly"]
__all__ = ["_PredictProbaMixin"]

import numpy as np
import pandas as pd

from sktime.datatypes import convert_to


class _PredictProbaMixin:
    """Mixin class for defaults in probabilistic prediction.

    Implements default probabilistic prediction methods to ensure that
    all methods work when one of the methods is implemented.

    Default dispatch order:

    _predict_interval -> _predict_quantiles, _predict_proba
    _predict_quantiles -> _predict_interval, _predict_proba
    _predict_var -> _predict_proba, _predict_interval
    _predict_proba -> _predict_var and _predict
    """

    def _predict_interval(self, coverage, **kwargs):
        """Compute/return prediction interval forecasts.

        private _predict_interval containing the core logic,
            called from predict_interval and default _predict_quantiles

        Parameters
        ----------
        kwargs : dict of proba prediction arguments
        coverage : float or list, optional (default=0.95)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_quantiles or implements_proba or implements_var

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for interval are:
        # default to quantiles if any of the other three methods are implemented

        # we default to _predict_quantiles if that is implemented or _predict_proba
        # since _predict_quantiles will default to _predict_proba if it is not
        alphas = []
        for c in coverage:
            # compute quantiles corresponding to prediction interval coverage
            #  this uses symmetric predictive intervals
            alphas.extend([0.5 - 0.5 * float(c), 0.5 + 0.5 * float(c)])

        # compute quantile forecasts corresponding to upper/lower
        pred_int = self._predict_quantiles(alpha=alphas, **kwargs)

        # change the column labels (multiindex) to the format for intervals
        # idx returned by _predict_quantiles is
        #   2-level MultiIndex with variable names, alpha
        # idx returned by _predict_interval should be
        #   3-level MultiIndex with variable names, coverage, lower/upper
        int_idx = self._get_columns(method="predict_interval", coverage=coverage)
        pred_int.columns = int_idx

        return pred_int

    def _predict_quantiles(self, alpha, **kwargs):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        kwargs : dict of proba prediction arguments
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_proba = self._has_implementation_of("_predict_proba")
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_interval or implements_proba or implements_var

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for quantiles are:
        # 1. default to interval if interval implemented
        # 2. default to proba if proba or var are implemented

        if implements_interval:
            pred_int = pd.DataFrame()
            for a in alpha:
                # compute quantiles corresponding to prediction interval coverage
                #  this uses symmetric predictive intervals:
                coverage = abs(1 - 2 * a)

                # compute quantile forecasts corresponding to upper/lower
                pred_a = self._predict_interval(coverage=[coverage], **kwargs)
                pred_int = pd.concat([pred_int, pred_a], axis=1)

            # now we need to subset to lower/upper depending
            #   on whether alpha was < 0.5 or >= 0.5
            #   this formula gives the integer column indices giving lower/upper
            col_selector_int = (np.array(alpha) >= 0.5) + 2 * np.arange(len(alpha))
            col_selector_bool = np.isin(np.arange(2 * len(alpha)), col_selector_int)
            num_var = len(pred_int.columns.get_level_values(0).unique())
            col_selector_bool = np.tile(col_selector_bool, num_var)

            pred_int = pred_int.iloc[:, col_selector_bool]
            # change the column labels (multiindex) to the format for intervals
            # idx returned by _predict_interval is
            #   3-level MultiIndex with variable names, coverage, lower/upper
            # idx returned by _predict_quantiles should be
            #   is 2-level MultiIndex with variable names, alpha
            int_idx = self._get_columns(method="predict_quantiles", alpha=alpha)
            pred_int.columns = int_idx

        elif implements_proba or implements_var:
            pred_proba = self.predict_proba(**kwargs)
            pred_int = pred_proba.quantile(alpha=alpha)

        return pred_int

    def _predict_var(self, cov=False, **kwargs):
        """Compute/return variance forecasts.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        kwargs : dict of proba prediction arguments
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            If cov=False:
                Column names are exactly those of ``y`` passed in ``fit``/``update``.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        from scipy.stats import norm

        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_interval or implements_quantiles or implements_proba

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for var are:
        # 1. default to proba if proba implemented
        # 2. default to interval if interval or quantiles are implemented

        if implements_proba:
            # todo: this works only univariate now, need to implement multivariate
            pred_dist = self.predict_proba(**kwargs)
            pred_var = pred_dist.var()

            return pred_var

        # if has one of interval/quantile predictions implemented:
        #   we get quantile forecasts for first and third quartile
        #   return variance of normal distribution with that first and third quartile
        if implements_interval or implements_quantiles:
            pred_int = self._predict_interval(coverage=[0.5], **kwargs)
            var_names = pred_int.columns.get_level_values(0).unique()
            vars_dict = {}
            for i in var_names:
                pred_int_i = pred_int[i].copy()
                # compute inter-quartile range (IQR), as pd.Series
                iqr_i = pred_int_i.iloc[:, 1] - pred_int_i.iloc[:, 0]
                # dividing by IQR of normal gives std of normal with same IQR
                std_i = iqr_i / (2 * norm.ppf(0.75))
                # and squaring gives variance (pd.Series)
                var_i = std_i**2
                vars_dict[i] = var_i

            # put together to pd.DataFrame
            #   the indices and column names are already correct
            pred_var = pd.DataFrame(vars_dict)

        return pred_var

    # todo: does not work properly for multivariate or hierarchical
    #   still need to implement this - once interface is consolidated
    def _predict_proba(self, marginal=True, **kwargs):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        kwargs : dict of proba prediction arguments
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_interval or implements_quantiles or implements_var

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for interval are:
        # default to quantiles if any of the other three methods are implemented

        # if any of the above are implemented, predict_var will have a default
        #   we use predict_var to get scale, and predict to get location
        pred_var = self._predict_var(**kwargs)
        pred_std = np.sqrt(pred_var)
        pred_mean = self.predict(**kwargs)
        # ensure that pred_mean is a pd.DataFrame
        df_types = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        pred_mean = convert_to(pred_mean, to_type=df_types)
        # pred_mean and pred_var now have the same format

        # default is normal with predict as mean and pred_var as variance
        from sktime.proba.normal import Normal

        index = pred_mean.index
        columns = pred_mean.columns
        pred_dist = Normal(mu=pred_mean, sigma=pred_std, index=index, columns=columns)

        return pred_dist
