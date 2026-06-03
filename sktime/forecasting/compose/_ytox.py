# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements a reducer that uses subset of y as exogenous features."""

import pandas as pd

from sktime.forecasting.base import BaseForecaster

__author__ = ["fkiraly", "dhairya-motta"]
__all__ = ["YToXForecaster"]


class YToXForecaster(BaseForecaster):
    """Forecaster that uses predictions of one y-subset as exogenous features.

    In `fit`:
    1. forecaster_B is fit to forecast `y_subset_B` of `y`, using `X` as exogeneous.
    2. forecaster_A is fit to forecast the remaining variables of `y`, using `X` joined
       with `y_subset_B` as exogeneous variables.

    In `predict`:
    1. forecaster_B is used to forecast `y_subset_B` to `fh`, using `X` as exogeneous.
    2. forecaster_A is used to forecast the remaining variables of `y` to `fh`, using
       `X` joined with the forecasts from step 1 as exogeneous.

    Parameters
    ----------
    forecaster_A : sktime forecaster
        The forecaster to use for the remaining variables of `y`.
    forecaster_B : sktime forecaster
        The forecaster to use for the `y_subset_B` variables.
    y_subset_B : list of str, or list of int
        The subset of variables from `y` to be forecasted by `forecaster_B` and used as
        exogenous features for `forecaster_A`.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.forecasting.compose import YToXForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 3, 2, 1]})
    >>> forecaster = YToXForecaster(
    ...     forecaster_A=NaiveForecaster(),
    ...     forecaster_B=NaiveForecaster(),
    ...     y_subset_B=["B"]
    ... )
    >>> forecaster.fit(y, fh=[1, 2])
    YToXForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2])
    """

    _tags = {
        "authors": ["fkiraly", "dhairya-motta"],
        "capability:multivariate": True,
        "capability:exogenous": True,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
    }

    def __init__(self, forecaster_A, forecaster_B, y_subset_B):
        self.forecaster_A = forecaster_A
        self.forecaster_B = forecaster_B
        self.y_subset_B = y_subset_B

        super().__init__()

    def _fit(self, y, X, fh):
        """Fit forecaster to training data."""
        self.forecaster_A_ = self.forecaster_A.clone()
        self.forecaster_B_ = self.forecaster_B.clone()

        y_cols = list(y.columns)

        # Determine subset B columns
        if isinstance(self.y_subset_B[0], int):
            self.y_subset_B_cols_ = [y_cols[i] for i in self.y_subset_B]
        else:
            self.y_subset_B_cols_ = list(self.y_subset_B)

        # Remaining columns go to A
        self.y_subset_A_cols_ = [
            col for col in y_cols if col not in self.y_subset_B_cols_
        ]

        self.y_cols_ = y_cols  # to reconstruct the order in predict

        y_B = y[self.y_subset_B_cols_]
        y_A = y[self.y_subset_A_cols_]

        # Fit B
        self.forecaster_B_.fit(y=y_B, X=X, fh=fh)

        # Prepare X for A by adding y_B as exogenous features
        if X is not None:
            X_for_A = pd.concat([X, y_B], axis=1)
        else:
            X_for_A = y_B.copy()

        # Fit A only if there are remaining columns
        if len(self.y_subset_A_cols_) > 0:
            self.forecaster_A_.fit(y=y_A, X=X_for_A, fh=fh)
        else:
            self.forecaster_A_ = None

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        y_pred_B = self.forecaster_B_.predict(fh=fh, X=X)

        if X is not None:
            X_for_A = pd.concat([X, y_pred_B], axis=1)
        else:
            X_for_A = y_pred_B.copy()

        if self.forecaster_A_ is not None:
            y_pred_A = self.forecaster_A_.predict(fh=fh, X=X_for_A)
            # Recombine in original column order
            y_pred = pd.concat([y_pred_A, y_pred_B], axis=1)
            y_pred = y_pred[self.y_cols_]
        else:
            y_pred = y_pred_B

        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data."""
        y_B = y[self.y_subset_B_cols_]
        y_A = y[self.y_subset_A_cols_]

        self.forecaster_B_.update(y=y_B, X=X, update_params=update_params)

        if X is not None:
            X_for_A = pd.concat([X, y_B], axis=1)
        else:
            X_for_A = y_B.copy()

        if self.forecaster_A_ is not None:
            self.forecaster_A_.update(y=y_A, X=X_for_A, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.forecasting.naive import NaiveForecaster

        params1 = {
            "forecaster_A": NaiveForecaster(),
            "forecaster_B": NaiveForecaster(),
            "y_subset_B": [0],
        }
        params2 = {
            "forecaster_A": NaiveForecaster(strategy="mean"),
            "forecaster_B": NaiveForecaster(strategy="last"),
            "y_subset_B": [0],
        }
        return [params1, params2]
