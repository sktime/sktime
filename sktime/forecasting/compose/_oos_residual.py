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
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": True,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
    }

    def __init__(self, forecaster, cv=None):
        self.forecaster = forecaster
        self.cv = cv
        self._in_forecaster = None
        self._oos_forecaster = None

        super().__init__()

        self.set_tags(
            **{
                "capability:exogenous": self.forecaster.get_tag("capability:exogenous"),
                "capability:pred_int": self.forecaster.get_tag("capability:pred_int"),
                "capability:missing_values": self.forecaster.get_tag(
                    "capability:missing_values"
                ),
                "capability:categorical_in_X": self.forecaster.get_tag(
                    "capability:categorical_in_X"
                ),
            }
        )

    def _split_fh(self, fh=None):
        if fh is None:
            return None, None

        in_fh = (
            fh.to_in_sample(self.cutoff)
            if not fh.is_all_out_of_sample(self.cutoff)
            else None
        )
        oos_fh = (
            fh.to_out_of_sample(self.cutoff)
            if not fh.is_all_in_sample(self.cutoff)
            else None
        )

        return in_fh, oos_fh

    def _fit(self, y, X, fh):
        _, oos_fh = self._split_fh(fh)

        self._oos_forecaster = clone(self.forecaster)
        self._oos_forecaster.fit(y=y, X=X, fh=oos_fh)

    def _custom_predict(self, fh, X, method_name, **method_kwargs):
        """Predicts using the OosResidualsWrapper."""
        _y = self._y
        _X = self._X
        in_fh, oos_fh = self._split_fh(fh)

        # Prepare CV
        cv = self.cv
        if cv is None and in_fh is not None:
            from sktime.split import ExpandingWindowSplitter
            from sktime.utils.warnings import warn

            warn(
                "No cross-validation splitter (`cv`) was provided to "
                "`OosResidualsWrapper`. A default "
                "`ExpandingWindowSplitter(initial_window=2)` has been "
                "initialized. Ensure that the wrapped forecaster is "
                "compatible with this splitter, or pass a suitable `cv` "
                "object when constructing the wrapper.",
                category=UserWarning,
                obj=self,
            )

            cv = ExpandingWindowSplitter(initial_window=2)
        elif cv is not None:
            cv = check_cv(cv)

        # Prepare placeholder for predictions
        columns = self._get_columns(method=method_name, **method_kwargs)
        index = fh.to_absolute_index(self.cutoff)
        if isinstance(_y.index, pd.MultiIndex):
            index = pd.MultiIndex.from_product(
                _y.index.levels[:-1] + [index], names=_y.index.names
            )
        preds = pd.DataFrame(0.0, index=index, columns=columns)

        # Out-of-sample predictions
        if oos_fh is not None:
            method = getattr(self._oos_forecaster, method_name)
            pred = method(X=X, fh=oos_fh, **method_kwargs)
            preds.loc[pred.index] = pred.values

        # In-sample predictions using cross-validation
        if in_fh is not None:
            self._in_forecaster = clone(self.forecaster)
            method = getattr(self._in_forecaster, method_name)

            # fit on the first training window
            window, horizon = next(cv.split(_y))
            new_y = _y.iloc[window]
            new_X = _X.iloc[window] if _X is not None else None
            self._in_forecaster.fit(y=new_y, X=new_X, fh=cv.get_fh())

            # update on all training windows
            for window, horizon in cv.split(_y):
                new_y = _y.iloc[window]
                new_X = _X.iloc[window] if _X is not None else None
                new__X = _X.iloc[horizon] if _X is not None else None

                self._in_forecaster.update(y=new_y, X=new_X, update_params=True)
                pred = method(X=new__X, **method_kwargs)

                common_idx = pred.index.intersection(preds.index)
                preds.loc[common_idx] = pred.loc[common_idx].values

        return preds

    def _predict(self, fh, X):
        return self._custom_predict(fh=fh, X=X, method_name="predict")

    def _predict_interval(self, fh, X=None, coverage=0.9):
        return self._custom_predict(
            fh=fh, X=X, method_name="predict_interval", coverage=coverage
        )

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
