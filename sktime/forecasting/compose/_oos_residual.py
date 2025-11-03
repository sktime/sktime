# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of OosResidualsWrapper."""

__author__ = ["geetu040"]

import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_cv


class OosResidualsWrapper(BaseForecaster):
    """Out-of-sample residuals wrapper for forecasters."""

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "capability:insample": True,
        "capability:pred_int": False,
        "capability:pred_int:insample": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": True,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
        "fit_is_empty": False,
    }

    def __init__(self, forecaster, cv=None):
        self.forecaster = forecaster
        self.cv = cv
        self._in_sample_forecaster = None
        self._out_of_sample_forecaster = None

        super().__init__()

        self.set_tags(
            **{
                "capability:exogenous": self.forecaster.get_tag("capability:exogenous"),
                # "capability:pred_int": self.forecaster.get_tag("capability:pred_int"),
                "capability:missing_values": self.forecaster.get_tag(
                    "capability:missing_values"
                ),
                "capability:categorical_in_X": self.forecaster.get_tag(
                    "capability:categorical_in_X"
                ),
            }
        )

    def _fit(self, y, X, fh):
        if fh is not None and not fh.is_all_in_sample(self.cutoff):
            out_of_sample_fh = fh.to_out_of_sample(self.cutoff)
        else:
            out_of_sample_fh = None

        self._out_of_sample_forecaster = clone(self.forecaster)
        self._out_of_sample_forecaster.fit(y=y, X=X, fh=out_of_sample_fh)

    def _predict(self, fh, X):
        """Predicts using the OosResidualsWrapper."""
        _y = self._y
        _X = self._X

        # Split fh into in-sample and out-of-sample
        out_of_sample_fh = (
            fh.to_out_of_sample(self.cutoff)
            if not fh.is_all_in_sample(self.cutoff)
            else None
        )
        in_sample_fh = (
            fh.to_in_sample(self.cutoff)
            if not fh.is_all_out_of_sample(self.cutoff)
            else None
        )

        # Prepare CV
        cv = self.cv
        if cv is None and in_sample_fh is not None:
            from sktime.split import ExpandingWindowSplitter

            cv = ExpandingWindowSplitter(initial_window=2)
        elif cv is not None:
            cv = check_cv(cv)

        # Prepare placeholder for predictions
        index = fh.to_absolute_index(self.cutoff)
        if isinstance(_y.index, pd.MultiIndex):
            index = pd.MultiIndex.from_product(
                _y.index.levels[:-1] + [index], names=_y.index.names
            )
        preds = pd.DataFrame(pd.NA, index=index, columns=_y.columns)

        # Out-of-sample predictions
        if out_of_sample_fh is not None:
            pred = self._out_of_sample_forecaster.predict(X=X, fh=out_of_sample_fh)
            preds.loc[pred.index] = pred.values

        # In-sample predictions using cross-validation
        if in_sample_fh is not None:
            self._in_sample_forecaster = clone(self.forecaster)
            # fit on the first training window
            window, horizon = next(cv.split(_y))
            new_y = _y.iloc[window]
            new_X = _X.iloc[window] if _X is not None else None
            self._in_sample_forecaster.fit(y=new_y, X=new_X, fh=cv.get_fh())

            # update on all training windows
            for window, horizon in cv.split(_y):
                new_y = _y.iloc[window]
                new_X = _X.iloc[window] if _X is not None else None
                new__X = _X.iloc[horizon] if _X is not None else None

                self._in_sample_forecaster.update(y=new_y, X=new_X, update_params=False)
                pred = self._in_sample_forecaster.predict(X=new__X)

                common_idx = pred.index.intersection(preds.index)
                preds.loc[common_idx] = pred.loc[common_idx].values

        return preds

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create test instances of the estimator.
        """
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.split import ExpandingWindowSplitter

        # Basic test parameters
        params1 = {
            "forecaster": NaiveForecaster(strategy="drift"),
        }

        # More complex test parameters
        params2 = {
            "forecaster": make_reduction(
                estimator=LinearRegression(),
                strategy="recursive",
                window_length=2,
            ),
            "cv": ExpandingWindowSplitter(initial_window=4, step_length=1, fh=1),
        }

        return [params1, params2]
