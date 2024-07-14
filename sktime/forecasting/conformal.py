"""Implements simple conformal forecast intervals.

Code based partially on NaiveVariance by ilyasmoutawwakil.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["ConformalIntervals"]
__author__ = ["fkiraly", "bethrice44"]

from math import floor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from sktime.datatypes import MTYPE_LIST_SERIES, convert, convert_to
from sktime.datatypes._utilities import get_slice
from sktime.forecasting.base import BaseForecaster
from sktime.utils.warnings import warn


class ConformalIntervals(BaseForecaster):
    r"""Empirical and conformal prediction intervals.

    Implements empirical and conformal prediction intervals, on absolute residuals.
    Empirical prediction intervals are based on sliding window empirical quantiles.
    Conformal prediction intervals are implemented as described in [1]_.

    All intervals wrap an arbitrary forecaster, i.e., add probabilistic prediction
    capability to a given point prediction forecaster (first argument).

    method="conformal_bonferroni" is the method described in [1]_,
        where an arbitrary forecaster is used instead of the RNN.
    method="conformal" is the method in [1]_, but without Bonferroni correction.
        i.e., separate forecasts are made which results in H=1 (at all horizons).
    method="empirical" uses quantiles of relative signed residuals on training set,
        i.e., y_t+h^(i) - y-hat_t+h^(i), ranging over i, in the notation of [1]_,
        at quantiles 0.5-0.5*coverage (lower) and 0.5+0.5*coverage (upper),
        as offsets to the point prediction at forecast horizon h
    method="empirical_residual" uses empirical quantiles of absolute residuals
        on the training set, i.e., quantiles of epsilon-h (in notation [1]_),
        at quantile point (1-coverage)/2 quantiles, as offsets to point prediction

    Parameters
    ----------
    forecaster : estimator
        Estimator to which probabilistic forecasts are being added
    method : str, optional, default="empirical"
        "empirical": predictive interval bounds are empirical quantiles from training
        "empirical_residual": upper/lower are plusminus (1-coverage)/2 quantiles
            of the absolute residuals at horizon, i.e., of epsilon-h
        "conformal_bonferroni": Bonferroni, as in Stankeviciute et al
            Caveat: this does not give frequentist but conformal predictive intervals
        "conformal": as in Stankeviciute et al, but with H=1,
            i.e., no Bonferroni correction under number of indices in the horizon
    initial_window : float, int or None, optional (default=max(10, 0.1*len(y)))
        Defines the size of the initial training window
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include for the initial window for the train split.
        If int, represents the relative number of train samples in the initial window.
        If None, the value is set to the larger of 0.1*len(y) and 10
    sample_frac : float, optional, default=None
        value in range (0,1) corresponding to fraction of y index to calculate
        residuals matrix values for (for speeding up calculation)
    verbose : bool, optional, default=False
        whether to print warnings if windows with too few data points occur
    n_jobs : int or None, optional, default=1
        The number of jobs to run in parallel for fit.
        -1 means using all processors.

    References
    ----------
    .. [1] Kamile Stankeviciute, Ahmed M Alaa and Mihaela van der Schaar.
        Conformal Time Series Forecasting. NeurIPS 2021.

    Examples
    --------
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>> from sktime.forecasting.conformal import ConformalIntervals  # doctest: +SKIP
    >>> from sktime.forecasting.naive import NaiveForecaster  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = NaiveForecaster(strategy="drift")  # doctest: +SKIP
    >>> conformal_forecaster = ConformalIntervals(forecaster)  # doctest: +SKIP
    >>> conformal_forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    ConformalIntervals(...)
    >>> pred_int = conformal_forecaster.predict_interval()  # doctest: +SKIP

    recommended use of ConformalIntervals together with ForecastingGridSearch
    is by 1. first running grid search, 2. then ConformalIntervals on the tuned params
    otherwise, nested sliding windows will cause high compute requirement

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.conformal import ConformalIntervals
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.param_est.plugin import PluginParamsForecaster
    >>> # part 1 = grid search
    >>> cv = ExpandingWindowSplitter(fh=[1, 2, 3])  # doctest: +SKIP
    >>> forecaster = NaiveForecaster()  # doctest: +SKIP
    >>> param_grid = {"strategy" : ["last", "mean", "drift"]}  # doctest: +SKIP
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=forecaster,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ... )  # doctest: +SKIP
    >>> # part 2 = plug in results of grid search into conformal intervals estimator
    >>> conformal_with_fallback = ConformalIntervals(NaiveForecaster())
    >>> gscv_with_conformal = PluginParamsForecaster(
    ...     gscv,
    ...     conformal_with_fallback,
    ...     params={"forecaster": "best_forecaster"},
    ... )  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> gscv_with_conformal.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    PluginParamsForecaster(...)
    >>> y_pred_quantiles = gscv_with_conformal.predict_quantiles()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "bethrice44"],
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "X_inner_mtype": MTYPE_LIST_SERIES,
        "y_inner_mtype": MTYPE_LIST_SERIES,
    }

    ALLOWED_METHODS = [
        "empirical",
        "empirical_residual",
        "conformal",
        "conformal_bonferroni",
    ]

    def __init__(
        self,
        forecaster,
        method="empirical",
        initial_window=None,
        sample_frac=None,
        verbose=False,
        n_jobs=None,
    ):
        if not isinstance(method, str):
            raise TypeError(f"method must be a str, one of {self.ALLOWED_METHODS}")

        if method not in self.ALLOWED_METHODS:
            raise ValueError(
                f"method must be one of {self.ALLOWED_METHODS}, but found {method}"
            )

        self.forecaster = forecaster
        self.method = method
        self.verbose = verbose
        self.initial_window = initial_window
        self.sample_frac = sample_frac
        self.n_jobs = n_jobs
        self.forecasters_ = []

        super().__init__()

        tags_to_clone = [
            "requires-fh-in-fit",
            "ignores-exogeneous-X",
            "handles-missing-data",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.forecaster, tags_to_clone)

    def _fit(self, y, X, fh):
        self.fh_early_ = fh is not None
        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(y=y, X=X, fh=fh)

        if self.fh_early_:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y=y,
                X=X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
                sample_frac=self.sample_frac,
            )
        else:
            self.residuals_matrix_ = None

        return self

    def _predict(self, fh, X):
        return self.forecaster_.predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        self.forecaster_.update(y, X, update_params=update_params)

        if update_params and len(y.index.difference(self.residuals_matrix_.index)) > 2:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y,
                X,
                self.forecaster_,
                self.initial_window,
                self.sample_frac,
                update=True,
            )

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
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
        y_pred = self.predict(fh=fh, X=X)

        if not isinstance(self.residuals_matrix_, dict):
            return self._predict_interval_series(
                fh=fh,
                coverage=coverage,
                y_pred=y_pred,
            )

        # otherwise, we have a hierarchical/multiindex y
        y_pred = convert_to(y_pred, ["pd-multiindex", "pd_multiindex_hier"])

        y_pred_index = y_pred.index.droplevel(-1)

        pred_ints = {}
        for ix in y_pred_index:
            y_pred_ix = y_pred.loc[ix]
            pred_ints[ix] = self._predict_interval_series(
                fh=fh,
                coverage=coverage,
                y_pred=y_pred_ix,
            )
        pred_int = pd.concat(pred_ints, axis=0, keys=y_pred_index)
        return pred_int

    def _predict_interval_series(self, fh, coverage, y_pred):
        """Compute prediction intervals predict_interval for series scitype."""
        fh_relative = fh.to_relative(self.cutoff)
        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_idx = fh_absolute.to_pandas()

        if self.fh_early_:
            residuals_matrix = self.residuals_matrix_
        else:
            residuals_matrix = self._compute_sliding_residuals(
                y=self._y,
                X=self._X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
                sample_frac=self.sample_frac,
            )

        ABS_RESIDUAL_BASED = ["conformal", "conformal_bonferroni", "empirical_residual"]

        var_names = self._get_varnames()

        cols = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(index=fh_absolute_idx, columns=cols)
        for fh_ind, offset in zip(fh_absolute, fh_relative):
            resids = np.diagonal(residuals_matrix, offset=offset)
            resids = resids[~np.isnan(resids)]
            if len(resids) < 1:
                resids = np.array([0], dtype=float)
                warn(
                    "In ConformalIntervals, sample fraction too low for "
                    "computing residuals matrix, using zero residuals. "
                    "Try setting sample_frac to a higher value.",
                    obj=self,
                    stacklevel=2,
                )
            abs_resids = np.abs(resids)
            coverage2 = np.repeat(coverage, 2)
            if self.method == "empirical":
                quantiles = 0.5 + np.tile([-0.5, 0.5], len(coverage)) * coverage2
                pred_int_row = np.quantile(resids, quantiles)
            if self.method == "empirical_residual":
                quantiles = 0.5 - 0.5 * coverage2
                pred_int_row = np.quantile(abs_resids, quantiles)
            elif self.method == "conformal_bonferroni":
                alphas = 1 - coverage2
                quantiles = 1 - alphas / len(fh)
                pred_int_row = np.quantile(abs_resids, quantiles)
            elif self.method == "conformal":
                quantiles = coverage2
                pred_int_row = np.quantile(abs_resids, quantiles)

            pred_int.loc[fh_ind] = pred_int_row

        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")
        y_pred.index = fh_absolute_idx

        for col in cols:
            if self.method in ABS_RESIDUAL_BASED:
                sign = 1 - 2 * (col[2] == "lower")
            else:
                sign = 1
            pred_int[col] = y_pred + sign * pred_int[col]

        return pred_int.convert_dtypes()

    def _parse_initial_window(self, y, initial_window=None):
        n_samples = len(y)

        if initial_window is None:
            if int(floor(0.1 * n_samples)) > 10:
                initial_window = int(floor(0.1 * n_samples))
            elif n_samples > 10:
                initial_window = 10
            else:
                initial_window = n_samples - 1

        initial_window_type = np.asarray(initial_window).dtype.kind

        if (
            initial_window_type == "i"
            and (initial_window >= n_samples or initial_window <= 0)
            or initial_window_type == "f"
            and (initial_window <= 0 or initial_window >= 1)
        ):
            raise ValueError(
                f"initial_window={initial_window} should be either positive and smaller"
                f" than the number of samples {n_samples} or a float in the "
                "(0, 1) range"
            )

        if initial_window is not None and initial_window_type not in ("i", "f"):
            raise ValueError(f"Invalid value for initial_window: {initial_window}")

        if initial_window_type == "f":
            n_initial_window = int(floor(initial_window * n_samples))
        elif initial_window_type == "i":
            n_initial_window = int(initial_window)

        return n_initial_window

    def _compute_sliding_residuals(
        self, y, X, forecaster, initial_window, sample_frac, update=False
    ):
        """Compute sliding residuals used in uncertainty estimates.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            sktime compatible time series to use in computing residuals matrix
        X : pd.DataFrame
            sktime compatible exogeneous time series to use in forecasts
        forecaster : sktime compatible forecaster
            forecaster to use in computing the sliding residuals
        initial_window : float, int or None, optional (default=max(10, 0.1*len(y)))
            Defines the size of the initial training window
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include for the initial window for the train split.
            If int, represents the relative number of train samples in the
            initial window.
            If None, the value is set to the larger of 0.1*len(y) and 10
        sample_frac : float
            for speeding up computing of residuals matrix.
            sample value in range (0, 1) to obtain a fraction of y indices to
            compute residuals matrix for
        update : bool
            Whether residuals_matrix has been calculated previously and just
            needs extending. Default = False

        Returns
        -------
        residuals_matrix : pd.DataFrame, row and column index = y.index[initial_window:]
            [i,j]-th entry is signed residual of forecasting y.loc[j] from y.loc[:i],
            using a clone of the forecaster passed through the forecaster arg.
            if sample_frac is passed this will have NaN values for 1 - sample_frac
            fraction of the matrix
        """
        y = convert_to(y, ["pd.Series", "pd-multiindex", "pd_multiindex_hier"])

        # vectorize over multiindex if y is hierarchical
        if isinstance(y.index, pd.MultiIndex):
            y_index = y.index.droplevel(-1)

            residuals = {}
            for ix in y_index:
                if X is not None:
                    X_ix = X.loc[ix]
                else:
                    X_ix = None
                y_ix = y.loc[ix]
                residuals[ix] = self._compute_sliding_residuals(
                    y_ix, X_ix, forecaster, initial_window, sample_frac, update
                )
            return residuals

        n_initial_window = self._parse_initial_window(y, initial_window=initial_window)

        full_y_index = y.iloc[n_initial_window:].index

        residuals_matrix = pd.DataFrame(
            columns=full_y_index, index=full_y_index, dtype="float"
        )

        if update and hasattr(self, "residuals_matrix_") and not sample_frac:
            remaining_y_index = full_y_index.difference(self.residuals_matrix_.index)
            if len(remaining_y_index) != len(full_y_index):
                overlapping_index = pd.Index(
                    self.residuals_matrix_.index.intersection(full_y_index)
                ).sort_values()
                residuals_matrix.loc[overlapping_index, overlapping_index] = (
                    self.residuals_matrix_.loc[overlapping_index, overlapping_index]
                )
            else:
                overlapping_index = None
            y_index = remaining_y_index
        else:
            y_index = full_y_index
            overlapping_index = None

        if sample_frac:
            y_sample = y_index.to_series().sample(frac=sample_frac)
            if len(y_sample) > 2:
                y_index = y_sample

        def _get_residuals_matrix_row(forecaster, y, X, id):
            y_train = get_slice(y, start=None, end=id)  # subset on which we fit
            y_test = get_slice(y, start=id, end=None)  # subset on which we predict

            X_train = get_slice(X, start=None, end=id)
            X_test = get_slice(X, start=id, end=None)
            forecaster.fit(y_train, X=X_train, fh=y_test.index)
            # Append fitted forecaster to list for extending for update
            self.forecasters_.append({"id": str(id), "forecaster": forecaster})

            try:
                residuals = forecaster.predict_residuals(y_test, X_test)
            except IndexError:
                warn(
                    "Couldn't predict after fitting on time series of length"
                    f"{len(y_train)}.",
                    obj=forecaster,
                )
            return residuals

        all_residuals = Parallel(n_jobs=self.n_jobs)(
            delayed(_get_residuals_matrix_row)(forecaster.clone(), y, X, id)
            for id in y_index
        )
        for idx, id in enumerate(y_index):
            residuals_matrix.loc[id] = all_residuals[idx]

        if overlapping_index is not None:

            def _extend_residuals_matrix_row(y, X, id):
                forecasters_df = pd.DataFrame(self.forecasters_)
                forecaster_to_extend = forecasters_df.loc[
                    forecasters_df["id"] == str(id)
                ]["forecaster"].values[0]

                y_test = get_slice(y, start=id, end=None)
                X_test = get_slice(X, start=id, end=None)

                try:
                    residuals = forecaster_to_extend.predict_residuals(y_test, X_test)
                except IndexError:
                    warn(
                        f"Couldn't predict with existing forecaster for cutoff {id}"
                        " with existing forecaster.",
                        obj=forecaster,
                    )
                return residuals

            extend_residuals = Parallel(n_jobs=self.n_jobs)(
                delayed(_extend_residuals_matrix_row)(y, X, id)
                for id in overlapping_index
            )

            for idx, id in enumerate(overlapping_index):
                residuals_matrix.loc[id] = extend_residuals[idx]

        return residuals_matrix

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
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params1 = {"forecaster": FORECASTER}
        params2 = {"forecaster": FORECASTER, "method": "conformal", "sample_frac": 0.9}

        return [params1, params2]
